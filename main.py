import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp, PageSumDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
import logging
from nltk import sent_tokenize
from modeling_bart_ours import PageSumModel
from transformers import Adafactor
from config import arxiv, arxiv_discourse, pubmed, govreport, multinews

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 3)  # batch size
    args.epoch = getattr(args, 'epoch', 100)  # epoch
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 6)  # accumulate step
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn")  # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000)  # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0)  # gradient norm
    args.seed = getattr(args, "seed", 970903)  # random seed
    args.pretrained = getattr(args, "pretrained", None)  # pretrained model
    args.max_lr = getattr(args, "max_lr", 2e-3)  # max learning rate
    args.datatype = getattr(args, "datatype", "base")  # data type
    args.dataset = getattr(args, "dataset", "govreport")  # dataset
    args.smooth = getattr(args, "smooth", 0.1)  # label smoothing
    args.length_penalty = getattr(args, "length_penalty", 2.0)  # length penalty
    args.do_generate = getattr(args, "do_generate", False)  # do generate
    args.page_max_len = getattr(args, "page_max_len", 1024)  # max length for one page
    args.tgt_max_len = getattr(args, "tgt_max_len", 1024)  # max length for target
    args.gen_max_len = getattr(args, "gen_max_len", 900)  # max length for generate
    args.gen_min_len = getattr(args, "gen_min_len", 500)  # min length for generate
    args.num_clusters = getattr(args, "num_pages", 7)  # number of pages
    args.optim = getattr(args, "optim", "adam")  # optimizer
    args.page_type = getattr(args, "page_type", None)  # cluster type, (None or 'multi_doc')
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", True)  # gradient checkpointing


class label_smoothing_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smoothing_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, input, target):
        input = input.transpose(1, 2) # [batch_size, seq_len, word_num]
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = torch.mul(loss, mask).sum() / mask.sum()
        return loss


def evaluation(args):
    # load data
    if args.config == "arxiv":
        arxiv(args)
    elif args.config == "arxiv_discourse":
        arxiv_discourse(args)
    elif args.config == "pubmed":
        pubmed(args)
    elif args.config == "govreport":
        govreport(args)
    elif args.config == "multinews":
        multinews(args)
    else:
        base_setting(args)
    tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = PageSumDataset(f"./{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, page_max_len=args.page_max_len, tgt_max_len=args.tgt_max_len, num_pages=args.num_pages, page_type=args.page_type)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = PageSumModel.from_pretrained(model_path, num_layers=args.num_layers, use_cache=True, gradient_checkpointing=False)
    if args.cuda:
        scorer = scorer.cuda()

    scorer.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    scorer.eval()

    model_name = args.model_pt.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    print(model_name)
    root_dir = "./result/%s"%model_name
    mkdir(root_dir)
    
    cnt = 0
    loss = 0

    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1, rouge2 = 0, 0
    scorer.set_seq_num(args.num_clusters)
    do_generate = True
    do_score = False

    with torch.no_grad(), open(os.path.join(root_dir, f"test.out"), "w") as f:
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            input_ids = batch["src_input_ids"]
            input_ids = input_ids.view(input_ids.size(0), -1)
            input_mask = input_ids != tok.pad_token_id
            if do_generate:
                summaries = scorer.generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2,
                    early_stopping=False,
                    seq_num=args.num_clusters,
                )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for (sample, d) in zip(batch["data"], dec):
                    sents = sent_tokenize(d)
                    score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                    _rouge1 = score["rouge1"].fmeasure
                    _rouge2 = score["rouge2"].fmeasure
                    rouge1 += _rouge1
                    rouge2 += _rouge2
                    cnt += 1
                for x in dec:
                    x = x.replace("\n", " ")
                    print(x, file=f)
            if i % 10 == 0:
                print(f"batch: {i}")
            
            if do_score:
                decoder_input_ids = batch["tgt_input_ids"]
                decoder_attention_mask = decoder_input_ids != tok.pad_token_id
                output = scorer(
                    input_ids=input_ids, 
                    attention_mask=input_mask,
                    decoder_input_ids=decoder_input_ids, 
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                    )
                output = output[0]
                output = output[:, :-1]
                gold = decoder_input_ids[:, 1:]
                loss += mle_fn(output.transpose(1, 2), gold)
            
    print(loss / i)
    print(rouge1 / cnt)
    print(rouge2 / cnt)


def test(dataloader, scorer, args, gpuid, tok):
    scorer.eval()
    loss = 0
    cnt = 0
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1, rouge2 = 0, 0
    if args.do_generate:
        with torch.no_grad():
            for (i, batch) in enumerate(dataloader):
                if args.cuda:
                    to_cuda(batch, args.gpuid[0])
                input_ids = batch["src_input_ids"]
                input_ids = input_ids.view(input_ids.size(0), -1)
                input_mask = input_ids != tok.pad_token_id
                if len(args.gpuid) > 1:
                    summaries = scorer.module.generate(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        early_stopping=True,
                        seq_num=args.num_clusters
                    )
                else:
                    summaries = scorer.generate(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        early_stopping=True,
                        seq_num=args.num_clusters
                    )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for (sample, d) in zip(batch["data"], dec):
                    sents = sent_tokenize(d)
                    score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                    rouge1 += score["rouge1"].fmeasure
                    rouge2 += score["rouge2"].fmeasure
                    cnt += 1
                if i % 100 == 0:
                    print(f"batch: {i}")
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        if len(args.gpuid) > 1:
            rouge1 = torch.FloatTensor([rouge1]).to(gpuid)
            dist.all_reduce(rouge1, op=dist.reduce_op.SUM)
            rouge1 = rouge1.item() / len(args.gpuid)
            rouge2 = torch.FloatTensor([rouge2]).to(gpuid)
            dist.all_reduce(rouge2, op=dist.reduce_op.SUM)
            rouge2 = rouge2.item() / len(args.gpuid)
        scorer.train()
        return rouge1, rouge2
    else:
        with torch.no_grad():
            for (i, batch) in enumerate(dataloader):
                if args.cuda:
                    to_cuda(batch, gpuid)
                input_ids = batch["src_input_ids"]
                input_ids = input_ids.view(input_ids.size(0), -1)
                input_mask = input_ids != tok.pad_token_id
                decoder_input_ids = batch["tgt_input_ids"]
                decoder_attention_mask = decoder_input_ids != tok.pad_token_id
                output = scorer(
                    input_ids=input_ids, 
                    attention_mask=input_mask,
                    decoder_input_ids=decoder_input_ids, 
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=False
                    )
                output = output[0]
                output = output[:, :-1]  # truncate last token
                gold = batch["tgt_input_ids"][:, 1:]  # shift right
                loss += mle_fn(output.transpose(1, 2), gold)
                if i % 100 == 0:
                    print(f"batch: {i}")
                cnt += 1
        loss = loss / cnt
        if len(args.gpuid) > 1:
            loss = torch.FloatTensor([loss]).to(gpuid)
            dist.all_reduce(loss, op=dist.reduce_op.SUM)
            loss = loss.item() / len(args.gpuid)
        scorer.train()
        return loss


def run(rank, args):
    if args.config == "arxiv":
        arxiv(args)
    elif args.config == "arxiv_discourse":
        arxiv_discourse(args)
    elif args.config == "pubmed":
        pubmed(args)
    elif args.config == "govreport":
        govreport(args)
    elif args.config == "multinews":
        multinews(args)
    else:
        base_setting(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache")) + 2
        recorder = Recorder(id, args.log)
    tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = PageSumDataset(f"./{args.dataset}/{args.datatype}/train", args.model_type, page_max_len=args.page_max_len, tgt_max_len=args.tgt_max_len, num_pages=args.num_pages, page_type=args.page_type)
    val_set = PageSumDataset(f"./{args.dataset}/{args.datatype}/val", args.model_type, is_test=True, page_max_len=args.page_max_len, tgt_max_len=args.tgt_max_len, num_pages=args.num_pages, page_type=args.page_type)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = PageSumModel.from_pretrained(model_path, gradient_checkpointing=args.gradient_checkpointing, use_cache=not args.gradient_checkpointing)
    if len(args.model_pt) > 0:
        scorer.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if len(args.gpuid) == 1:
            scorer = scorer.cuda()
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            scorer = nn.parallel.DistributedDataParallel(scorer.to(gpuid), [gpuid], find_unused_parameters=False)
    scorer.train()
    mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    init_lr = args.max_lr / args.warmup_steps
    if args.optim == "adafactor":
        s_optimizer = Adafactor(scorer.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    else:
        s_optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    if is_master:
        recorder.write_config(args, [scorer], __file__)
    minimum_loss = 1e5
    all_step_cnt = 0
    if len(args.gpuid) > 1:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    if is_mp:
        scorer.module.set_seq_num(args.num_clusters)
    else:
        scorer.set_seq_num(args.num_clusters)
    # start training
    for epoch in range(args.epoch):
        s_optimizer.zero_grad()
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            input_ids = batch["src_input_ids"]
            input_ids = input_ids.view(input_ids.size(0), -1)
            input_mask = input_ids != tok.pad_token_id
            decoder_input_ids = batch["tgt_input_ids"]
            decoder_attention_mask = decoder_input_ids != tok.pad_token_id
            output = scorer(
                input_ids=input_ids, 
                attention_mask=input_mask,
                decoder_input_ids=decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=False
                )
            output = output[0]
            output = output[:, :-1]  # truncate last token
            gold = batch["tgt_input_ids"][:, 1:]  # shift right
            loss = mle_fn(output.transpose(1, 2), gold)
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                print("id: %d"%id)
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f"%(epoch+1, epoch_step, avg_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_loss = 0
            del loss, output

            if all_step_cnt % 1000 == 0 and all_step_cnt != 0 and step_cnt == 0:
                if args.do_generate:
                    rouge1, rouge2 = test(val_dataloader, scorer, args, gpuid, tok)
                    loss = 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
                else:
                    loss = test(val_dataloader, scorer, args, gpuid, tok)
                if loss < minimum_loss and is_master:
                    minimum_loss = loss
                    if is_mp:
                        recorder.save(scorer.module, "scorer_best.bin")
                    else:
                        recorder.save(scorer, "scorer_best.bin")
                    recorder.print("best loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    if is_mp:
                        recorder.save(scorer.module, "scorer.bin")
                    else:
                        recorder.save(scorer, "scorer.bin")
                    recorder.save(s_optimizer, "optimizer.bin")
                    recorder.print("val loss: %.6f"%loss)
                    if args.do_generate:
                        recorder.print(f"ROUGE-1: {rouge1:.6f}, ROUGE-2: {rouge2:.6f}")


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate")
    parser.add_argument("-l", "--log", action="store_true", help="log")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="base", type=str, help="config path")
    args = parser.parse_args()
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
