from got10k.experiments import ExperimentGOT10k
import os
from got10k.utils.convert import got2newformat

got2newformat(
    src_json_path="E:/eval-vis/GOT10K/SiamFDB.json",  # 官网下载的旧版文件
    dst_json_path="E:/eval-vis/GOT10K/SiamFDB_new.json"  # 转换后的新版文件
)

# 评价结果所在路径
# result_dir = 'E:/eval-vis/GOT10K'

# 选择要作图的跟踪器名称
# tracker_names = ['ours', 'SiamFDB', 'AFSN', 'ATOM', 'DaSiamRPN', 'ECO', 'MDNet', 'SiamCAR', 'SiamFC', 'SiamRPN++', 'SPM']
#
# # 生成每个跟踪器的 JSON 文件路径列表
# report_files = [os.path.join(result_dir, f"{name}.json") for name in tracker_names]
report_files = ['E:/eval-vis/GOT10K/SiamFDB_new.json']

# 检查文件是否存在（可选）
for file in report_files:
    if not os.path.exists(file):
        print(f"警告：文件 {file} 不存在！")

# 设置实验参数
experiment = ExperimentGOT10k('D:/GOT-10k_test', subset='test')

# 作图
experiment.plot_curves(report_files, tracker_names)