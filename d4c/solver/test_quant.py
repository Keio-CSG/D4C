import sys
sys.path.append("/home/zhang/Project/D4C")
import argparse
import json
import os
import sys
from copy import copy, deepcopy
import random
import torch

from clip_benchmark.datasets.builder import (build_dataset, dataset_collection,
                                             get_dataset_collate_fn,
                                             get_dataset_collection_from_file,
                                             get_dataset_default_task)
from clip_benchmark.metrics import (captioning, image_caption_selection,
                                    linear_probe, zeroshot_classification,
                                    zeroshot_retrieval)
from clip_benchmark.models import MODEL_TYPES, load_clip

from d4c.solver.utils import parse_config, convert_weights, world_info_from_env, get_cali_data, save_images_to_file
from d4c.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all, set_ch_axis
from d4c.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from d4c.quantization.observer import ObserverBase
from d4c.quantization.quant_block import specials
from recon import reconstruction
import time
import torch.nn as nn
import json
from d4c.quantization.fake_quant import QuantizeBase
from d4c.solver.dfq_utils import gen_rand_img, train_fake_img_d4c, train_fake_img_baseline


def get_parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="cifar10", nargs="+", help="Dataset to use for the benchmark.")
    parser.add_argument('--dataset_root', default="/data2/user/zhang/clip/clip_benchmark", type=str, help="Dataset root folder where the datasets are downloaded.")
    parser.add_argument('--split', type=str, default="test", help="Dataset split to use.")
    parser.add_argument('--test_split', dest="split", action='store', type=str, default="test", help="Dataset split to use.")
    parser.add_argument('--train_split', type=str, nargs="+", default="train", help="Dataset train split name.")
    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument('--val_split', default=None, type=str, nargs="+", help="Dataset validation split name. Mutually exclusive with val_proportion.")
    mutually_exclusive.add_argument('--val_proportion', default=None, type=float, nargs="+", help="What is the share of the train dataset will be used for validation part, if it doesn't predefined. Mutually exclusive with val_split.")

    parser.add_argument('--model', type=str, nargs="+", default="ViT-B/32", help="Model architecture to use.")
    parser.add_argument('--task', type=str, default="zeroshot_classification", help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser.add_argument('--no_amp', action="store_false", dest="amp", default=True, help="Whether to use mixed precision.")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--recall_k', default=[5], type=int, help="For retrieval, select the k for Recall@K metric.", nargs="+")
    parser.add_argument('--fewshot_k', default=-1, type=int, help="For linear probe, how many shots. -1 = whole dataset.")
    parser.add_argument('--fewshot_epochs', default=10, type=int, help="For linear probe, how many epochs.")
    parser.add_argument('--fewshot_lr', default=0.1, type=float, help="For linear probe, what is the learning rate.")
    parser.add_argument("--distributed", action="store_true", help="Evaluation in parallel.")
    parser.add_argument('--seed', default=0, type=int, help="Random seed.")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--normalize', default=True, type=bool, help="Features normalization.")
    parser.add_argument('--model_cache_dir', default=None, type=str, help="Directory to where downloaded models are cached")
    parser.add_argument('--feature_root', default="features", type=str, help="Feature root folder where the features are stored.")
    parser.add_argument('--annotation_file', default="", type=str, help="Text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser.add_argument('--custom_classname_file', default=None, type=str, help="Use custom json file with classnames for each dataset, where keys are dataset names and values are list of classnames.")
    parser.add_argument('--custom_template_file', default=None, type=str, help="Use custom json file with prompts for each dataset, where keys are dataset names and values are list of prompts. For instance, to use CuPL prompts, use --custom_template_file='cupl_prompts.json'.")
    parser.add_argument('--dump_classnames', default=False, action="store_true", help="Dump classnames to the results json file.")
    parser.add_argument('--dump_templates', default=False, action="store_true", help="Dump templates to the results json file.")
    parser.add_argument('--output', default="result.json", type=str, help="Output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_{model}_{task}.json'")
    parser.add_argument('--quiet', dest='verbose', action="store_false", help="Suppress verbose messages.")
    parser.add_argument('--save_clf', default=None, type=str, help="Optionally save the classification layer output by the text tower.")
    parser.add_argument('--load_clfs', nargs='+', default=[], type=str, help="Optionally load and average mutliple layers output by text towers.")
    parser.add_argument('--skip_existing', default=False, action="store_true", help="Whether to skip an evaluation if the output file exists.")
    parser.add_argument('--model_type', default="openai_clip", type=str, choices=MODEL_TYPES, help="Clip model type.")
    parser.add_argument('--wds_cache_dir', default=None, type=str, help="Optional cache directory for webdataset only.")

    parser.add_argument('--fp_model', action='store_true', default=False, help="Run with FP model.")
    parser.add_argument('--q_config', type=str, default='./exp/config66.yaml', help="Quantization configuration.")
    parser.add_argument('--recon', action='store_true', default=False, help="Optimize quantization parameters via model reconstruction.")
    parser.add_argument('--custom_file', default='./d4c/template/custom_classnames_templates.json', type=str, help="Custom class names and templates for generation and calibration.")
    parser.add_argument('--dfq', action='store_true', default=False, help="Enable data-free quantization.")
    parser.add_argument('--gen_img', action='store_true', default=False, help="Generate fake image by D4C.")
    parser.add_argument('--gen_method', type=str, default='d4c', help="Use baseline BN/PSE method or D4C to generate image. Select from d4c and baseline.")
    parser.add_argument('--d4c_config', type=int, default=3, help="Select D4C configuration for image generation. 0: PGSI; 1: PGSI + SCG; 2: PGSI + PAE; 3: Method PGSI + SCG + PAE.")
    parser.add_argument('--gen_batch_size', type=int, default=16, help="Batch size for fake image generation in mini-batches.")
    parser.add_argument('--gen_lr', type=float, default=0.01, help="Learning rate for fake image generation.")
    parser.add_argument('--gen_iter', type=int, default=3000, help="Iterations for fake image generation.")
    parser.add_argument('--img_path', default=None, help="Save path of generated images. Default is a None.")

    args = parser.parse_args()
    return parser, args


def main(base):
    q_config = parse_config(base.q_config)
    # Use model given by --model args
    model = _as_list(base.model)[0]

    # Ge list of dataset to evaluate on
    dataset_list = _as_list(base.dataset)
    if os.path.isfile(dataset_list[0]):
        datasets = get_dataset_collection_from_file(dataset_list[0])
        dataset = datasets[0]
    elif dataset_list[0] in dataset_collection:
        dataset = dataset_collection[dataset_list[0]][0]
    else:
        dataset = dataset_list[0]
    train_split = _as_list(base.train_split)[0]
    val_split = _as_list(base.val_split)[0] if base.val_split is not None else None
    val_proportion = _as_list(base.val_proportion)[0] if base.val_proportion is not None else None

    if base.verbose:
        print(f"Model: {model}")
        print(f"Dataset: {dataset}")

    args = copy(base)
    args.model = model
    args.dataset = dataset
    args.train_split = train_split
    args.val_split = val_split
    args.val_proportion = val_proportion
    args.q_config = q_config

    run(args)


def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l


def run(args):
    """Console script for clip_benchmark."""
    if torch.cuda.is_available():
        if args.distributed:
            local_rank, rank, world_size = world_info_from_env()
            device = f'cuda:{local_rank}'
            torch.cuda.set_device(device)
        else:
            device = "cuda"
        args.device = device
    else:
        args.device = "cpu"
    # set seed.
    torch.manual_seed(args.seed)

    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    else:
        dataset_name = args.dataset
    dataset_slug = dataset_name.replace('/', '_')
    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))

    with open(args.custom_file, 'r') as f:
        custom_list = json.load(f)

    if args.model in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']:
        is_transformer = True
    elif args.model in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
        is_transformer = False
    else:
        raise NotImplementedError("Model name is not valid.")

    model, transform, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=None,
        cache_dir=args.model_cache_dir,
        device=args.device
    )

    model = convert_weights(model)

    cali_dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform,
        split=args.train_split,
        annotation_file=args.annotation_file,
        download=True,
        task=args.task,
        custom_template_file=args.custom_template_file,
        wds_cache_dir=args.wds_cache_dir,
    )
    collate_fn_cal = get_dataset_collate_fn(args.dataset)
    if args.dataset.startswith("wds/"):
        cali_dataloader = torch.utils.data.DataLoader(
            cali_dataset.batched(args.q_config.calibrate.visual), batch_size=None,
            shuffle=True, num_workers=args.num_workers,
        )
    else:
        cali_dataloader = torch.utils.data.DataLoader(
            cali_dataset, batch_size=args.q_config.calibrate.visual,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=collate_fn_cal
        )

    eval_dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform,
        split=args.split,
        annotation_file=args.annotation_file,
        download=True,
        task=args.task,
        custom_template_file=args.custom_template_file,
        custom_classname_file=args.custom_classname_file,
        wds_cache_dir=args.wds_cache_dir,
    )
    collate_fn_eval = get_dataset_collate_fn(args.dataset)
    if args.verbose:
        try:
            print(f"Dataset size: {len(eval_dataset)}")
        except TypeError:
            print("IterableDataset has no len()")
        print(f"Dataset split: {args.split}")
        if hasattr(eval_dataset, "classes") and eval_dataset.classes:
            try:
                print(f"Dataset classes: {eval_dataset.classes}")
                print(f"Dataset number of classes: {len(eval_dataset.classes)}")
            except AttributeError:
                print("Dataset has no classes.")

    if args.dataset.startswith("wds/"):
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset.batched(args.batch_size), batch_size=None,
            shuffle=False, num_workers=args.num_workers,
        )
    else:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn_eval
        )

    output = args.output.format(
        model=args.model,
        task=args.task,
        dataset=dataset_slug,
    )
    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return
    if args.verbose:
        print(f"Running '{args.task}' on '{dataset_name}' with the model '{args.model}'")

    zeroshot_templates = eval_dataset.templates if hasattr(eval_dataset, "templates") else None
    if args.verbose:
        print(f"Zero-shot templates: {zeroshot_templates}")
    classnames = eval_dataset.classes if hasattr(eval_dataset, "classes") else None
    assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"

    custom_classnames = custom_list.get('classnames', [])

    if not args.fp_model:
        cali_data = get_cali_data(cali_dataloader, args.q_config.calibrate.visual)
        cali_data = cali_data.to(device)

        cali_templates = cali_dataset.templates
        if args.verbose:
            print(f"Calibrate templates: {cali_templates}")

        cali_text = []
        for classname in custom_classnames:
            texts = [template.format(c=classname) for template in cali_templates]
            cali_text.extend(texts)
        random.shuffle(cali_text)
        cali_text = cali_text[:args.q_config.calibrate.text]
        cali_text = tokenizer(cali_text).to(device)

        if args.dfq:
            cali_data = gen_rand_img(cali_data)
            cali_data = cali_data.to(args.device)

            if args.gen_img:
                gen_text = []
                gen_templates = custom_list.get('templates', [])
                for classname in custom_classnames:
                    texts = [template.format(c=classname) for template in gen_templates]
                    gen_text.extend(texts)
                gen_text = tokenizer(gen_text).to(device)

                gen_st = time.time()
                if args.gen_method == 'baseline':
                    train_fake_img_baseline(model, cali_data, is_transformer, args.gen_iter, args.gen_batch_size, args.gen_lr, args.device)
                elif args.gen_method == 'd4c':
                    train_fake_img_d4c(model, cali_data, gen_text, args.d4c_config, args.gen_iter, args.gen_batch_size, args.gen_lr, args.device)
                else:
                    raise NotImplementedError("Specify method from d4c or baseline.")
                gen_ed = time.time()
                print('Pseudo image generation time: {}'.format(gen_ed - gen_st))

            if args.img_path is not None:
                save_images_to_file(cali_data, save_path=args.img_path)

        model = quantize_model(model, args.q_config)
        model.cuda()
        model.eval()

        fp_model = deepcopy(model)
        disable_all(fp_model)
        for name, module in model.named_modules():
            if isinstance(module, ObserverBase):
                module.set_name(name)

        # Set Activation Quantizers to Per-Channel and 8-bit in MLP of Text Encoder
        # Use Per-Channel Quantization for QKV and Lin1 Projection where the Parameters Can be Folded into LN
        for block_name, block in model.transformer.named_modules():
            if isinstance(block, QuantizedBlock) and 'mlp' in block_name:
                set_ch_axis(block.c_proj)
                for _, layer in block.named_modules():
                    if isinstance(layer, QuantizeBase):
                        layer.set_bit(8)
        for block_name, block in model.named_modules():
            if isinstance(block, QuantizedBlock) and 'attn' in block_name:
                for proj in [block.q_proj, block.k_proj, block.v_proj]:
                    set_ch_axis(proj)
        for block_name, block in model.named_modules():
            if isinstance(block, QuantizedBlock) and 'mlp' in block_name:
                set_ch_axis(block.c_fc)

        with torch.no_grad():
            st = time.time()
            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            model(cali_data[:32], cali_text[:32])
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            model(cali_data[:32], cali_text[:32])
            ed = time.time()
            print('Calibration time: {}'.format(ed - st))

        if args.recon:
            enable_quantization(model)

            for name, quant_module in model.visual.named_modules():
                if isinstance(quant_module, (QuantizedLayer, QuantizedBlock)):
                    quant_module.branch = 'visual'
            for name, quant_module in model.transformer.named_modules():
                if isinstance(quant_module, (QuantizedLayer, QuantizedBlock)):
                    quant_module.branch = 'text'

            def recon_model(module: nn.Module, fp_module: nn.Module, block_recon: True):
                """
                Block reconstruction.
                """
                if block_recon:
                    target = (QuantizedLayer, QuantizedBlock)
                else:
                    target = QuantizedLayer
                for name, child_module in module.named_children():
                    if isinstance(child_module, target):
                        print('begin reconstruction for module:\n{}'.format(str(child_module)))
                        reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_text, cali_data, args.q_config.recon)
                    else:
                        recon_model(child_module, getattr(fp_module, name), block_recon)

            # Start reconstruction
            if is_transformer:
                recon_model(model.visual, fp_model.visual, block_recon=False)
            else:
                recon_model(model.visual, fp_model.visual, block_recon=True)

            recon_model(model.transformer, fp_model.transformer, block_recon=False)

        enable_quantization(model)

        for n, m in model.named_modules():
            if hasattr(m, 'drop_prob'):
                m.drop_prob = 1

    metrics = zeroshot_classification.evaluate(
        model,
        eval_dataloader,
        tokenizer,
        classnames, zeroshot_templates,
        device=args.device,
        amp=args.amp,
        verbose=args.verbose,
        save_clf=args.save_clf,
        load_clfs=args.load_clfs,
    )

    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "task": args.task,
        "metrics": metrics,
    }
    if hasattr(eval_dataset, "classes") and eval_dataset.classes and args.dump_classnames:
        dump["classnames"] = eval_dataset.classes
    if hasattr(eval_dataset, "templates") and eval_dataset.templates and args.dump_templates:
        dump["templates"] = eval_dataset.templates
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


def quantize_model(model, config_quant):

    if type(model) in specials:
        return specials[type(model)](model, config_quant.w_qconfig, config_quant.a_qconfig)

    def replace_module(module, w_qconfig, a_qconfig):
        for name, child in module.named_children():
            if type(child) in specials:
                setattr(module, name, specials[type(child)](child, w_qconfig, a_qconfig))
            else:
                replace_module(child, w_qconfig, a_qconfig)

    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig)

    return model


if __name__ == "__main__":
    parser, base = get_parser_args()
    sys.exit(main(base))  # pragma: no cover
