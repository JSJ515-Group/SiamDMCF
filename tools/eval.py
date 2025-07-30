import os.path
# import sys
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset, GOT10kDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from toolkit.visualization import draw_success_precision, draw_eao, draw_f1
import matplotlib.pyplot as plt

# sys.path.append("./")

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=10)


parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
parser.add_argument('--dataset_dir', type=str, default='/home/yxy/gxy/testing_dataset',
                    help='dataset root directory')
parser.add_argument('--dataset', type=str, default='UAV123',
                    help='dataset name')
parser.add_argument('--tracker_result_dir', type=str, default='../tools/results', help='tracker result root')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--vis', dest='vis', action='store_true')
parser.add_argument('--show_video_level', dest='show_video_level', action='store_true')
parser.add_argument('--num', type=int, help='number of processes to eval', default=5)
args = parser.parse_args()


def main():
    tracker_dir = os.path.join(args.tracker_result_dir, args.dataset)
    trackers = glob(os.path.join(args.tracker_result_dir,
                                 args.dataset,
                                 args.tracker_prefix + '*'))
    trackers = [x.split('\\')[-1] for x in trackers]

    root = os.path.join(args.dataset_dir, args.dataset)

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'LaSOT' in args.dataset or 'LASOT' in args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret,
                                       norm_precision_ret=norm_precision_ret)
    elif 'UAV' in args.dataset or 'UAV123' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       video=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif args.dataset in ['VOT2016', 'VOT2018', "VOT2019"]:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                                                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        benchmark.show_result(eao_result)
        ar_benchmark.show_result(ar_result, eao_result,
                                 show_video_level=args.show_video_level)
        if args.vis:
            draw_eao(eao_result)
    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                              show_video_level=args.show_video_level)
        if args.vis:
            draw_f1(f1_result)
    elif 'GOT' in args.dataset:
        dataset = GOT10kDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)


if __name__ == '__main__':
    main()
