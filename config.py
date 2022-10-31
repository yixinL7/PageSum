def arxiv(args):
    args.batch_size = getattr(args, 'batch_size', 1)  # batch size
    args.epoch = getattr(args, 'epoch', 10)  # epoch
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 16)  # accumulate step
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn")  # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000)  # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0)  # gradient norm
    args.seed = getattr(args, "seed", 970903)  # random seed
    args.pretrained = getattr(args, "pretrained", None)  # pretrained model
    args.max_lr = getattr(args, "max_lr", 2e-3)  # max learning rate
    args.datatype = getattr(args, "datatype", "base")  # data type
    args.dataset = getattr(args, "dataset", "arxiv")  # dataset
    args.smooth = getattr(args, "smooth", 0.1)  # label smoothing
    args.length_penalty = getattr(args, "length_penalty", 2.0)  # length penalty
    args.do_generate = getattr(args, "do_generate", False)  # do generate
    args.page_max_len = getattr(args, "page_max_len", 1024)  # max length for one page
    args.tgt_max_len = getattr(args, "tgt_max_len", 400)  # max length for target
    args.gen_max_len = getattr(args, "gen_max_len", 350)  # max length for generate
    args.gen_min_len = getattr(args, "gen_min_len", 100)  # min length for generate
    args.num_pages = getattr(args, "num_pages", 7)  # number of pages
    args.optim = getattr(args, "optim", "adam")  # optimizer
    args.page_type = getattr(args, "page_type", None)  # cluster type, (None or 'multi_doc')
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)  # gradient checkpointing


def arxiv_discourse(args):
    args.batch_size = getattr(args, 'batch_size', 1)  # batch size
    args.epoch = getattr(args, 'epoch', 10)  # epoch
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 16)  # accumulate step
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn")  # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000)  # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0)  # gradient norm
    args.seed = getattr(args, "seed", 970903)  # random seed
    args.pretrained = getattr(args, "pretrained", None)  # pretrained model
    args.max_lr = getattr(args, "max_lr", 2e-3)  # max learning rate
    args.datatype = getattr(args, "datatype", "section")  # data type
    args.dataset = getattr(args, "dataset", "arxiv")  # dataset
    args.smooth = getattr(args, "smooth", 0.1)  # label smoothing
    args.length_penalty = getattr(args, "length_penalty", 2.0)  # length penalty
    args.do_generate = getattr(args, "do_generate", False)  # do generate
    args.page_max_len = getattr(args, "page_max_len", 1024)  # max length for one page
    args.tgt_max_len = getattr(args, "tgt_max_len", 400)  # max length for target
    args.gen_max_len = getattr(args, "gen_max_len", 350)  # max length for generate
    args.gen_min_len = getattr(args, "gen_min_len", 100)  # min length for generate
    args.num_pages = getattr(args, "num_pages", 8)  # number of pages
    args.optim = getattr(args, "optim", "adam")  # optimizer
    args.page_type = getattr(args, "page_type", "multi_doc")  # cluster type, (None or 'multi_doc')
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)  # gradient checkpointing


def pubmed(args):
    args.batch_size = getattr(args, 'batch_size', 1)  # batch size
    args.epoch = getattr(args, 'epoch', 10)  # epoch
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 16)  # accumulate step
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn")  # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000)  # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0)  # gradient norm
    args.seed = getattr(args, "seed", 970903)  # random seed
    args.pretrained = getattr(args, "pretrained", None)  # pretrained model
    args.max_lr = getattr(args, "max_lr", 2e-3)  # max learning rate
    args.datatype = getattr(args, "datatype", "base")  # data type
    args.dataset = getattr(args, "dataset", "pubmed")  # dataset
    args.smooth = getattr(args, "smooth", 0.1)  # label smoothing
    args.length_penalty = getattr(args, "length_penalty", 2.0)  # length penalty
    args.do_generate = getattr(args, "do_generate", False)  # do generate
    args.page_max_len = getattr(args, "page_max_len", 1024)  # max length for one page
    args.tgt_max_len = getattr(args, "tgt_max_len", 400)  # max length for target
    args.gen_max_len = getattr(args, "gen_max_len", 400)  # max length for generate
    args.gen_min_len = getattr(args, "gen_min_len", 150)  # min length for generate
    args.num_pages = getattr(args, "num_pages", 7)  # number of pages
    args.optim = getattr(args, "optim", "adam")  # optimizer
    args.page_type = getattr(args, "page_type", None)  # cluster type, (None or 'multi_doc')
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)  # gradient checkpointing


def govreport(args):
    args.batch_size = getattr(args, 'batch_size', 2)  # batch size
    args.epoch = getattr(args, 'epoch', 10)  # epoch
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 8)  # accumulate step
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
    args.num_pages = getattr(args, "num_pages", 20)  # number of pages
    args.optim = getattr(args, "optim", "adafactor")  # optimizer
    args.page_type = getattr(args, "page_type", None)  # cluster type, (None or 'multi_doc')
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", True)  # gradient checkpointing


def multinews(args):
    args.batch_size = getattr(args, 'batch_size', 1)  # batch size
    args.epoch = getattr(args, 'epoch', 10)  # epoch
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 16)  # accumulate step
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn")  # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000)  # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0)  # gradient norm
    args.seed = getattr(args, "seed", 970903)  # random seed
    args.pretrained = getattr(args, "pretrained", None)  # pretrained model
    args.max_lr = getattr(args, "max_lr", 2e-3)  # max learning rate
    args.datatype = getattr(args, "datatype", "base")  # data type
    args.dataset = getattr(args, "dataset", "multinews")  # dataset
    args.smooth = getattr(args, "smooth", 0.1)  # label smoothing
    args.length_penalty = getattr(args, "length_penalty", 2.0)  # length penalty
    args.do_generate = getattr(args, "do_generate", False)  # do generate
    args.page_max_len = getattr(args, "page_max_len", 1024)  # max length for one page
    args.tgt_max_len = getattr(args, "tgt_max_len", 400)  # max length for target
    args.gen_max_len = getattr(args, "gen_max_len", 400)  # max length for generate
    args.gen_min_len = getattr(args, "gen_min_len", 150)  # min length for generate
    args.num_pages = getattr(args, "num_pages", 7)  # number of pages
    args.optim = getattr(args, "optim", "adam")  # optimizer
    args.page_type = getattr(args, "page_type", "multi_doc")  # cluster type, (None or 'multi_doc')
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)  # gradient checkpointing