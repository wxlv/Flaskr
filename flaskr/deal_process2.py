#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/10 9:19
# @Author  : Frederick
# @email   :frederick_hx@163.com

# pip install sklearn
import os
from matplotlib import animation
import warnings
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.stats.multicomp import MultiComparison  # 多重方差分析
from statsmodels.stats.anova import anova_lm
from sklearn import metrics
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.stats.anova as anova
import dython
import matplotlib.pyplot as pyplot
import pandas as pd
from scipy.stats import chi2
import numpy as np
import matplotlib
from scipy.stats import kstest
from sklearn.manifold import TSNE
from sklearn import preprocessing
# 加入以下代码：避免plot图例中的中文显示乱码
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
# import tushare as ts
# 使用方差分析
# 导入kmeans模块和轮廓系数
warnings.filterwarnings('ignore')
# 显示所有列(参数设置为None代表显示所有行，也可以自行设置数字)
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置数据的显示长度，默认为50
pd.set_option('max_colwidth', 200)
# 禁止自动换行(设置为Flase不自动换行，True反之)
pd.set_option('expand_frame_repr', False)

# 数据预处理


def deal_data(data):
    # 131碘-甲亢治疗次数大量缺失，采用均值填充
    data['jiakang_times'].fillna(data["jiakang_times"].mean(), inplace=True)

    # hesu_nums(甲)核素药-碘[131I]数量，因大量缺失，采用均值填充
    data['hesu_nums'].fillna(data["hesu_nums"].mean(), inplace=True)

    # smoking_years，吸烟年数，因大量缺失，采用均值填充
    data['smoking_years'].fillna(data["smoking_years"].mean(), inplace=True)

    # drinking_years，喝酒年数，因大量缺失，采用均值填充
    data['drinking_years'].fillna(data["drinking_years"].mean(), inplace=True)

    # ABO_bloodType，ABO血型，因大量缺失，采用均值填充
    data['ABO_bloodType'].fillna(data["ABO_bloodType"].mean(), inplace=True)

    # Rh_bloodType，Rh血型，因大量缺失，采用均值填充
    data['Rh_bloodType'].fillna(data["Rh_bloodType"].mean(), inplace=True)
    # print(data.info())
    return data
#
#   # 方差分析的结果我们需要看P值，本例中P值等于 0.0小于0.05，说明处理间存在显著差异，具体哪个处理间存在差异还需要通过多重检验来看。与前一种方法也相同。
#   # 多重分析 比较常用的检验方法是邓肯多重检验（Tukey HSD test）
#   mc = MultiComparison(df_melt['Value'], df_melt['Class'])
#   tukey_result = mc.tukeyhsd(alpha=0.5)
#   print(tukey_result)
#   # https://zyicu.cn/?p=3290。网址

# 多因素方差分析
# 首先要确定好哪个是y值（应变量），哪些个是自变量


def multi_variance_analysis(data):
    formula = 'gender~C(age)+C(ABO_bloodType)+C(Rh_bloodType)+C(days)+C(admissionMethod)+C(leaveMethod)' \
              '+C(paymentType)+C(bloodTransfusion_flag)+C(operation_flag)' \
              '+C(jiakang_medicine_flag)+C(jiakang_times)+C(hesu_nums)+C(bichao_flag)' \
              '+C(total_cost)+C(total_cost_selfpay)+C(gene_medical_service_fee)+C(gene_treatment_operation_fee)+' \
              '+C(nursing_fee)+C(other_exp_compre_medical_services)+C(pathological_diagnosis_fee)+' \
              'C(lab_diagnosis_fee)+C(imag_diagnosis_fee)+C(clinical_diagnosis_pro_fee)+' \
              'C(non_surgical_treatment_program_fees)+C(clinical_physiotherapy_fee)+C(surgical_treatment_fee)+' \
              'C(anesthesia_fee)+C(surgery_fee)+C(rehabilitation_fee)+C(chinese_medicine_treatment_fee)+' \
              'C(western_medicine_fee)+C(antibacterial_drug_fee)+C(proprietary_chinese_medicine_fee)+C(chinese_herbal_medicine_fee)' \
              '+C(blood_cost)+C(albumin_products_fee)+C(globulin_products_fee)+C(coagulation_factor_products)' \
              '+C(cytokine_product_fee)+C(disp_medical_materials_examination)+C(disp_medical_materials_treatment)' \
              '+C(disp_medical_materials_surgery)+C(other_fees)+C(smoking_flag)+C(smoking_years)+C(drinking_flag)+C(drinking_years)+C(age):C(drinking_years)'
    anova_results = anova_lm(ols(formula, data, type=48).fit())
    # print(anova_results)
    # https://wap.sciencenet.cn/blog-907836-1337256.html


# 4.多变量之间的相关性（散点图矩阵）
def correalation():
    corr = data.loc[:, ["gender", "age", "ABO_bloodType", "Rh_bloodType", "days", "admissionMethod", "leaveMethod", "paymentType", "bloodTransfusion_flag", "operation_flag", "jiakang_medicine_flag",
                        "jiakang_times", "hesu_nums", "bichao_flag", "total_cost", "total_cost_selfpay", "gene_medical_service_fee",
                        "gene_treatment_operation_fee", "nursing_fee", "other_exp_compre_medical_services", "pathological_diagnosis_fee", "lab_diagnosis_fee",
                        "imag_diagnosis_fee", "clinical_diagnosis_pro_fee", "non_surgical_treatment_program_fees",
                        "clinical_physiotherapy_fee", "surgical_treatment_fee", "anesthesia_fee", "surgery_fee",
                        "rehabilitation_fee", "chinese_medicine_treatment_fee", "western_medicine_fee", "antibacterial_drug_fee", "proprietary_chinese_medicine_fee",
                        "chinese_herbal_medicine_fee", "blood_cost", "albumin_products_fee", "globulin_products_fee",
                        "coagulation_factor_products", "cytokine_product_fee", "disp_medical_materials_examination",
                        "disp_medical_materials_treatment", "disp_medical_materials_surgery", "other_fees", "smoking_flag", "smoking_years", "drinking_flag", "drinking_years"]].corr()
    print("corr\n", corr)

# ‘胳膊肘方法’确定k值


def get_k(data):
    '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 30):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(data)
        SSE.append(estimator.inertia_)
    X = range(1, 30)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()

# “轮廓系数法”确定K值


def get_k2(data):
    Scores = []  # 存放轮廓系数
    for k in range(2, 30):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(data)  # 此时跑的数据是740183条归一化之后的数据
        Scores.append(silhouette_score(
            data, estimator.labels_, metric='euclidean'))
    X = range(2, 30)
    plt.xlabel('k')
    plt.ylabel('轮廓系数')
    plt.plot(X, Scores, 'o-')
    plt.show()

# 计算评价指标


def get_evaluation_index(true_label, pred_label):
    # 1.AdjustedRandIndex调整兰德系数：兰德系数是一种指标，互信息是一种指标，经过调整得到调整兰德系数和调整互信息两种指标。
    # 调整的意义在于：对于随机聚类，分值应该尽量低。
    AdjustedRandIndex_score = metrics.cluster.adjusted_rand_score(
        np.array(true_label), np.array(pred_label))
    print("AdjustedRandIndex_score:", AdjustedRandIndex_score)

    # 2.Mutual Information based scores 互信息
    NMI_score = metrics.adjusted_mutual_info_score(
        np.array(true_label), np.array(pred_label))
    print("NMI_score:", NMI_score)

    # 3.V-measure
    V_measure_score = metrics.v_measure_score(
        np.array(true_label), np.array(pred_label))
    print("V_measure_score:", V_measure_score)

    # 4.FMI指数，越接近1越好
    FMI_score = metrics.fowlkes_mallows_score(
        np.array(true_label), np.array(pred_label))
    print("FMI_score:", FMI_score)

# t-SNE计算


def prepare_tsne(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = TSNE(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(
        n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    return df_matrix

# 可视化


def plot_animation(df, label_column, name):
    def update(num):
        ax.view_init(200, num)

    N = 360
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_3d_df['x'], tsne_3d_df['y'], tsne_3d_df['z'], c=tsne_3d_df[label_column],
               s=6, depthshade=True, cmap='Paired')
    ax.set_zlim(-15, 25)
    ax.set_xlim(-20, 20)
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, N, blit=False, interval=50)
    ani.save('{}.gif'.format(name), writer='imagemagick')
    plt.show()

# 计算相关性分析


def plot_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

# PCA计算


def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(
        n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    return df_matrix

# 画出选择最好的n_components：累积可解释方差贡献率曲线


def draw_pca(data):
    pca_line = PCA().fit(data)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
              45, 46, 47, 48], np.cumsum(pca_line.explained_variance_ratio_))
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48])  # 这是为了限制坐标轴显示为整数
    # 让x轴上面的数字进行倾斜60度显示
    # 这里是调节横坐标的倾斜度，rotation是度数
    plt.xticks(rotation=30)
    plt.xlabel("number of components after dimension reduction")
    plt.ylabel("cumulative explained variance ratio")
    plt.show()

# 聚类分析


def cluster_analysis(data):
    # Prepare models
    kmeans = KMeans(n_clusters=4, init='k-means++').fit(data)
    normalized_vectors = preprocessing.normalize(data)
    normalized_kmeans = KMeans(n_clusters=4).fit(normalized_vectors)
    # min_samples = data.shape[1] + 1
    # dbscan = DBSCAN(eps=0.0001, min_samples=2).fit(data) # DBSCAN聚类
    # Print results
    print('kmeans: {}'.format(silhouette_score(
        data, kmeans.labels_, metric='euclidean')))
    print('Cosine kmeans:{}'.format(silhouette_score(normalized_vectors,
                                                     normalized_kmeans.labels_,
                                                     metric='cosine')))
    # print('DBSCAN: {}'.format(silhouette_score(data, dbscan.labels_, metric='cosine')))
    return kmeans, normalized_kmeans

# 聚类结果可视化


def cluster_isualizatio(data, kmeans, normalized_kmeans):
    pca_df = prepare_pca(3, data, kmeans.labels_)
    sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set1")
    plt.show()

    # normalized kmeans聚类结果降维及可视化
    pca_df = prepare_pca(3, data, normalized_kmeans.labels_)
    sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set2")
    plt.show()

    # DBSCAN聚类结果降维及可视化
    # pca_df = prepare_pca(3, data, dbscan.labels_)
    # sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set3")
    # plt.show()


# 读取指定路径下的文件


def find_file(search_path, include_str=None, filter_strs=None):
    """
    查找指定目录下所有的文件（不包含以__开头和结尾的文件）或指定格式的文件，若不同目录存在相同文件名，只返回第1个文件的路径
    :param search_path: 查找的目录路径
    :param include_str: 获取包含字符串的名称
    :param filter_strs: 过滤包含字符串的名称
    """
    if filter_strs is None:
        filter_strs = []

    files = []
    # 获取路径下所有文件
    names = os.listdir(search_path)
    for name in names:
        path = os.path.abspath(os.path.join(search_path, name))
        if os.path.isfile(path):
            # 如果不包含指定字符串则
            if include_str is not None and include_str not in name:
                continue

            # 如果未break，说明不包含filter_strs中的字符
            for filter_str in filter_strs:
                if filter_str in name:
                    break
            else:
                files.append(path)
        else:
            files += find_file(path, include_str=include_str,
                               filter_strs=filter_strs)
    return files

# Shapiro-Wilk正态性检验


def sw(data):
    S, p = stats.shapiro(data)
    # print("111:",p)
    if p <= 0.05:
        return "<0.0001"
    else:
        return round(p, 4)

# Kolmogorov-Smirnov正态性检验


def ks(data):
    ks_stat = kstest(data, 'norm')
    # print("ks_stat:",ks_stat[1])
    if ks_stat[1] <= 0.05:
        return "<0.0001"

# 3.描述性统计分析
# 只传入一列数据


def describe_analysis(data):
    # 1.统计个数:总频数
    col_count = data.shape[0]

    # 2.有效频数（%）
    notnull_count = data.notnull().sum()
    notnull_ratio = notnull_count / col_count * 100

    # 3.缺失值（%）
    null_count = data.isnull().sum()
    null_ratio = null_count / col_count * 100

    # 4.均值
    mean_value = round(data.mean(), 2)

    # 5.标准差
    std_value = round(data.std(), 2)

    # 6.最小值
    min_value = data.min()

    # 7.Q1的值
    xx = dict(data.describe(percentiles=[0.25, 0.50, 0.75]))
    # print("xx:",xx)
    Q1 = xx['25%']

    # 8.中位数 .median()
    median_value = data.median()

    # 9.Q3的值
    Q3 = xx['75%']

    # 10.最大值
    max_value = data.max()

    # 11.Shapiro-Wilk正态性检验
    # Shapiro-Wilk正态性检验：Shapiro-Wilk正态性检验的p值<=0.05，
    # 拒绝原假设，说明该列变量不服从正太分布。通常适用于样本量小于2000的情况。
    p_sw = sw(data)

    # 12.Kolmogorov-Smirnov正态性检验
    # Kolmogorov-Smirnov正态性检验：Kolmogorov-Smirnov正态性检验的p值<=0.05，
    # 拒绝原假设，说明该列变量不服从正态分布。通常适用于样本量大于2000的情况。
    p_ks = ks(data)

    return col_count, notnull_count, notnull_ratio, null_count, null_ratio, mean_value, std_value, min_value, Q1, median_value, Q3, max_value, p_sw, p_ks

# 4.0 计算两组数据的p值
# 连接：https://blog.csdn.net/weixin_32412379/article/details/116192973


def get_p_value(arrA, arrB):
    # 注意：计算P值的时候，把nan填充为0值
    # print(arrB)
    a = list(np.array(arrA))
    # print("a:",a)
    b = list(np.array(arrB))
    # print("b:",b)
    t, p = stats.ttest_ind(a, b)
    return p

# 4.0 计算列联系数[不确定！！！]
# https://www.jianshu.com/p/0e8ebdfc889d


def lielian_xishu(col1, col2):
    M = np.array([list(np.array(col1)), list(np.array(col2))])
    # print("M:",M)

    a = np.array(M.sum(axis=0))
    b = np.array(M.sum(axis=1))
    e = 0
    t = a.sum()

    for i in range(0, len(a)):
        for j in range(0, len(b)):
            # print([M[i][j], a[i], b[j]])
            e += M[j][i] * M[j][i] / (a[i] * b[j])

    d = t * (e - 1)
    c = round(np.sqrt(d / (t + d)), 4)  # 列联系数
    return c


# 4.相关性分析
def correalation_analysis(col1, col2):
    # 1.统计个数:总频数
    col_count = col1.shape[0]

    # 2.有效频数（%）
    notnull_count = col1.notnull().sum()
    notnull_ratio = notnull_count / col_count * 100
    # print("notnull_count:", notnull_count)
    # print("notnull_ratio:", notnull_ratio)

    # 3.Cramer'V系数
    # 注意：源码中都只是返回一个V系数，而没有p值！！！！
    # Cramer'V系数==0.0，因为Cramer'V系数<0.5，说明第一列与第二列相关性较弱
    v_xishu = round(dython.nominal.cramers_v(col1, col2), 4)  # 保留4位有效数字

    # 3.1 对缺失值用均值填充
    col1.fillna(col1.mean(), inplace=True)
    col2.fillna(col2.mean(), inplace=True)
    p_value = round(get_p_value(col1, col2), 4)  # 保留4位有效数字

    # 4.列联系数
    # 列联系数==0.3148，因为列联系数<0.5,说明第一列与第二列弱相关
    ll_xs = lielian_xishu(col1, col2)
    p_value2 = round(get_p_value(col1, col2), 4)  # 保留4位有效数字

    return col_count, notnull_count, notnull_ratio, v_xishu, p_value, ll_xs, p_value2

# 5.0 eta系数


def ANOVA_eta(df, a=0.05):
    from scipy.stats import f
    '''
    进行单因素方差分析
    输入值：df - pd.DataFrame，第一列为水平，第二列为观测值；a - 显著性水平，默认为0.05
    返回类型：字典
    返回值：方差分析相关数据
    '''
    res = {'SSA': 0, 'SST': 0}
    mu = df[df.columns[1]].mean()
    da = df.groupby(df.columns[0]).agg({df.columns[1]: ['mean', 'count']})
    da.columns = ['mean', 'count']
    res['df_A'] = len(list(da.index)) - 1  # 自由度
    # 组间误差平方和
    for row in da.index:
        res['SSA'] += (da.loc[row, 'mean'] - mu) ** 2 * da.loc[row, 'count']
    # 总误差平方和
    for e in df[df.columns[1]].values:
        res['SST'] += (e - mu) ** 2
    res['SSE'] = res['SST'] - res['SSA']  # 组内误差平方和
    res['df_E'] = len(df) - res['df_A'] - 1  # 残差自由度
    res['df_T'] = len(df) - 1  # 总和自由度
    res['MSA'] = res['SSA'] / res['df_A']  # 组间均方
    res['MSE'] = res['SSE'] / res['df_E']  # 组内均方
    res['F'] = res['MSA'] / res['MSE']  # F值
    res['p_value'] = 1 - f(res['df_A'], res['df_E']).cdf(res['F'])  # p值
    res['a'] = a
    res['F_alpha'] = f(res['df_A'], res['df_E']).ppf(1 - a)  # 基于显著性水平a的F临界值
    # https://zhuanlan.zhihu.com/p/137779235
    # eta公式：eta平方 = SSA / SST
    res['eta'] = math.sqrt(res['SSA'] / res['SST'])  # eta系数
    return round(res['F'], 4), res['df_T'], round(res['p_value'], 4), round(res['eta'], 4)

# 5.eta相关性分析


def eta_corr(col1, col2):
    # 1.统计个数:总频数
    col_count = col1.shape[0]

    # 2.有效频数（%）
    notnull_count = col1.notnull().sum()
    notnull_ratio = notnull_count / col_count * 100

    # 3.eta相关系数
    from scipy.stats import f
    df = pd.concat(objs=[col1, col2], axis=1)
    # print("df:",df)
    F_value, df_value, p_values, eta = ANOVA_eta(df, a=0.05)
    return col_count, notnull_count, notnull_ratio, F_value, df_value, p_values, eta

# 6.0 处理有效频数


def youxiao_pinshu(length, nums):
    # 2.有效频数（%）
    notnull_ratio = nums / nums * 100
    return round(notnull_ratio, 2)

# 6.1 缺失值个数 = 总频数 - 有效频数


def queshizhi(count, youxiao):
    return count - youxiao

# 6.2 计算缺失率


def queshi_rate(queshizhi, count):
    return queshizhi / count

# 6.3 数字转字符串


def float_2_str(x):
    return str(x)

# 6.4 计算中位数


def median_q1_q3(data):
    xx = dict(data.describe(percentiles=[0.25, 0.50, 0.75]))
    print("hhh:", xx)
    Q1 = xx['25%']
    Q2 = xx['50%']
    Q3 = xx['75%']
    return Q1, Q2, Q3

# 6.单因素分析


def single_analysis(data, col1, col2):
    # 1.统计该分组下的次数count
    res = data.groupby([col1]).size().reset_index(name='Count')
    # print(res)

    # 2.某列的长度个数
    length = data.shape[0]

    # 3.有效频数（%）
    # res['有效频数'] = res.loc[:, :].apply(lambda x: youxiao_pinshu(length,x["Count"]), axis=1)
    res['有效频数'] = res["Count"]
    res['有效率（%）'] = res.loc[:, :].apply(
        lambda x: youxiao_pinshu(length, x["Count"]), axis=1)

    # 数字转字符串拼接
    res['有效频数2'] = res.loc[:, :].apply(
        lambda x: float_2_str(x["有效频数"]), axis=1)
    res['有效率2（%）'] = res.loc[:, :].apply(
        lambda x: float_2_str(x["有效率（%）"]), axis=1)
    res['有效频数(%)'] = res['有效频数2'] + "("+res['有效率2（%）'] + "%" + ")"

    # 4.缺失值
    res["缺失值"] = res.loc[:, :].apply(
        lambda x: queshizhi(x["Count"], x["有效频数"]), axis=1)

    # 5.缺失率
    res["缺失率(%)"] = res.loc[:, :].apply(
        lambda x: queshi_rate(x["缺失值"], x["Count"]), axis=1)

    # 数字转字符串拼接
    res['缺失值2'] = res.loc[:, :].apply(lambda x: float_2_str(x["缺失值"]), axis=1)
    res['缺失率2(%)'] = res.loc[:, :].apply(
        lambda x: float_2_str(x["缺失率(%)"]), axis=1)
    res['缺失值(%)'] = res['缺失值2'] + "(" + res['缺失率2(%)'] + "%" + ")"

    del res["有效频数"]
    del res["有效率（%）"]
    del res["有效频数2"]
    del res["有效率2（%）"]
    del res["缺失值"]
    del res["缺失率(%)"]
    del res["缺失值2"]
    del res["缺失率2(%)"]
    # print("qqqqq:",res)

    # 6.分组统计均值，应该是统计col2（观察变量）的均值
    xxxxx = pd.DataFrame(data.groupby([col1])[col2].mean())
    y = xxxxx[[col2]].values
    mean_value = pd.DataFrame(y)
    mean_value.columns = ['均值']
    # print("分组均值：",mean_value)

    # # 拼接一起
    res3 = pd.concat([res, round(mean_value, 4)],
                     axis=1).reset_index(drop=True)
    # print("resw:", res3)

    # 7.标准差
    x_std = pd.DataFrame(data.groupby([col1])[col2].std())
    y_s = x_std[[col2]].values
    std_value = pd.DataFrame(y_s)
    std_value.columns = ['标准差']
    # print("分组均值：", std_value)

    # # 拼接一起
    res4 = pd.concat([res3, round(std_value, 4)],
                     axis=1).reset_index(drop=True)
    # print("resw:", res4)

    # 8.最小值
    x_min = pd.DataFrame(data.groupby([col1])[col2].min())
    y_m = x_min[[col2]].values
    min_value = pd.DataFrame(y_m)
    min_value.columns = ['最小值']
    # print("分组最小值：", min_value)

    # # 拼接一起
    res5 = pd.concat([res4, min_value], axis=1).reset_index(drop=True)
    # print("resw:", res5)

    # 9.Q1
    df1 = (data.groupby([col1])[col2].quantile(
        [0.25, 0.75]).unstack().rename(columns={0.25: 'Q1', 0.75: 'Q3'}))
    # print("df1:",df1)
    df1_q = df1['Q1'].values
    Q1_value = pd.DataFrame(df1_q)
    Q1_value.columns = ['Q1']

    # 拼接
    res6 = pd.concat([res5, Q1_value], axis=1).reset_index(drop=True)
    # print("resw:", res6)

    # 10.中位数
    m = data.groupby([col1])[col2].apply(np.median)
    # print("mmm:",m)
    y_median = m.values
    median_value = pd.DataFrame(y_median)
    median_value.columns = ['中位数']

    # # 拼接一起
    res7 = pd.concat([res6, median_value], axis=1).reset_index(drop=True)
    # print("resw:", res7)

    # 11.Q3
    df1_q = df1['Q3'].values
    Q3_value = pd.DataFrame(df1_q)
    Q3_value.columns = ['Q3']

    # 拼接
    res8 = pd.concat([res7, Q3_value], axis=1).reset_index(drop=True)
    # print("resw:", res8)

    # 12.最大值
    x_max = pd.DataFrame(data.groupby([col1])[col2].max())
    y_max = x_max[[col2]].values
    max_value = pd.DataFrame(y_max)
    max_value.columns = ['最大值']
    # # 拼接一起
    res9 = pd.concat([res8, max_value], axis=1).reset_index(drop=True)
    # print("resw:", res9)

    # 13.Shapiro-Wilk正态性检验【注意:对全列进行正态性检验！！】
    # Shapiro-Wilk正态性检验：Shapiro-Wilk正态性检验的p值<=0.05，
    # 拒绝原假设，说明该列变量不服从正太分布。通常适用于样本量小于2000的情况。
    p_sw = sw(data[col2])
    res9['Shapiro-Wilk正态性检验'] = p_sw
    # print("res9:",res9)

    # 14.Kolmogorov-Smirnov正态性检验
    # Kolmogorov-Smirnov正态性检验：Kolmogorov-Smirnov正态性检验的p值<=0.05，
    # 拒绝原假设，说明该列变量不服从正态分布。通常适用于样本量大于2000的情况。
    p_ks = ks(data[col2])
    res9['Kolmogorov-Smirnov正态性检验'] = p_ks

    # 15.单因素方差分析
    from scipy.stats import f
    # print("xss:",col1)
    # print("xqqqqs:",type(col2))
    df = pd.concat(objs=[data[col1], data[col2]], axis=1)

    # print("df:",df)
    F_value, df_value, p_values, eta = ANOVA_eta(df, a=0.05)
    str_F_df_p = 'F值:' + str(F_value) + "," + "df值:" + \
        str(df_value) + "," + "p值" + str(p_values)
    # print("str_F_df_p::",str_F_df_p)
    # print("F值：",F_value)
    # print("df值：",df_value)
    # print("p值：",p_values)
    res9['单因素方差分析(推荐)'] = str_F_df_p
    # print("res9:", res9)

    # 16.Kw秩和检验:df：自由度
    # https://blog.csdn.net/qq_33169259/article/details/126571364
    observed = data[[col1, col2]]
    xxv = stats.chi2_contingency(observed=observed)
    crit = round(stats.chi2.ppf(q=0.95, df=5), 4)  # 95置信水平 df = 自由度
    # print("卡方值：",crit)  # 临界值，拒绝域的边界 当卡方值大于临界值，则原假设不成立，备择假设成立
    pp_value = round(xxv[1], 4)
    dff_value = round(xxv[2], 4)
    # print("dff_value:",dff_value)
    # print("pp_value:",pp_value)
    # print("xxv:",xxv)
    kafang_df_p = "卡方值:" + str(crit) + "," + "df:" + \
        str(dff_value) + "," + "p值:" + str(pp_value)
    res9['Kw秩和检验'] = kafang_df_p
    # print("单因素方差分析：",res9)
    return res9

# 1.定义主函数


def run_main():
    # 1.读取数据
    # data = pd.read_csv("analysis_data2.csv", encoding='utf-8')

    # 获取全部文件
    # f = find_file(".")
    # print("f1:",f)

    # 获取包含指定字符的文件
    f = find_file(".", include_str=".csv")[0]
    # print("f2:",f)

    data = pd.read_csv(f, encoding='utf-8')
    # print("data:",data.head())

    # 2.去掉无关列
    # 将operationName(手术名称)去掉
    del data['operationName']

    # 自动删除方差为0的列
    # data = data.loc[:, data.nunique() != 1]
    # print(data.info())

    # ------------------- 描述性统计 ----------------------------------------
    # 3.描述性统计
    col = "age"  # 'leaveMethod' # 传入列名即可
    analysisData = data[col]
    print("描述性分析原始数据：",analysisData)
    col_count, notnull_count, notnull_ratio, null_count,\
        null_ratio, mean_value, std_value, min_value, Q1, median_value,\
        Q3, max_value, p_sw, p_ks = describe_analysis(analysisData)
    # print("总患者数：", col_count)
    # print("有效频数{}({}%):".format(notnull_count, notnull_ratio))
    # print("缺失值{}({}%):".format(null_count, null_ratio))
    # print("均值：", mean_value)
    # print("标准差：", std_value)
    # print("最小值：", min_value)
    # print("Q1:", Q1)
    # print("中位数：", median_value)
    # print("Q3:", Q3)
    # print("最大值：", max_value)
    # print('Shapiro-Wilk正态性检验:', p_sw)
    # print("Kolmogorov-Smirnov正态性检验:", p_ks)

    youxiao_pinshu_str = str(notnull_count) + \
        "(" + str(notnull_ratio) + "%" + ")"
    queshi_rate_str = str(null_count) + "(" + str(null_ratio) + "%" + ")"
    dataF1 = pd.DataFrame()
    resultDf = dataF1.append(
        pd.DataFrame({'总患者数': [col_count], '有效频数(%)': [youxiao_pinshu_str],
                      '缺失值(%)': [queshi_rate_str], '均值': [mean_value],
                      '标准差': [std_value], '最小值': [min_value],
                      'Q1': [Q1], '中位数': [median_value],
                      'Q3': [Q3], '最大值': [max_value],
                      'Shapiro-Wilk正态性检验': [p_sw],
                      'Kolmogorov-Smirnov正态性检验': [p_ks]}),
        ignore_index=True)
    print("resultDf:", end="\n")
    print(resultDf, end="\n")

    # ------------------- 描述性统计end ----------------------------------------

    print("=" * 100, end="\n")
    # --------------------------- Cramer_V和列联系数_相关性分析 ------------------------------------
    # 4.相关性分析
    colm1 = "days"
    colm2 = "ABO_bloodType"
    col_count2, notnull_count2, notnull_ratio2, v_xishu, p_value2, ll_xs, p_value2 = correalation_analysis(
        data[colm1], data[colm2])
    # print("总患者数：", col_count2)
    # print("有效频数{}({}%):".format(notnull_count2, notnull_ratio2))
    # print("Cramer_V系数:", v_xishu)
    # print("Cramer_V_p值:", p_value2)
    # print("列联系数：", ll_xs)
    # print("列联_p值:", p_value2)

    youxiao_pinshu_str2 = str(notnull_count2) + \
        "(" + str(notnull_ratio2) + "%" + ")"
    Cramer_V_str = "V系数:" + str(v_xishu) + " " + "p值:" + str(p_value2)
    lielian_str = "列联系数:" + str(ll_xs) + " " + "p值:" + str(p_value2)

    dataF2 = pd.DataFrame()
    resultDf2 = dataF2.append(
        pd.DataFrame({'总患者数': [col_count2], '有效频数(%)': [youxiao_pinshu_str2],
                      'Cramer-v系数': [Cramer_V_str], '列联系数': [lielian_str]}),
        ignore_index=True)
    print("resultDf2:", resultDf2)

    # --------------------------- Cramer_V和列联系数_相关性分析end ----------------------------------

    print("=" * 100, end="\n")
    # --------------------------- eta相关性分析 --------------------------------------
    # https://zhuanlan.zhihu.com/p/422777890
    # 5.eta相关性分析
    colm3 = "days"
    colm4 = "ABO_bloodType"
    col_count3, notnull_count3, notnull_ratio3, F_value, df_value, p_values, eta = eta_corr(
        data[colm3], data[colm4])
    # print("总患者数:", col_count3)
    # print("有效频数{}({}%):".format(notnull_count3, notnull_ratio3))
    # print("F_value:", F_value)
    # print("df_value:", df_value)
    # print("p_values:", p_values)
    # print("eta:", eta)

    youxiao_pinshu_str3 = str(notnull_count3) + \
        "(" + str(notnull_ratio3) + "%" + ")"
    Eta_str = "F:" + str(F_value) + " " + "df:" + str(df_value) + \
        " " + "p值:" + str(p_values) + " " + "Eta系数:" + str(eta)

    dataF3 = pd.DataFrame()
    resultDf3 = dataF3.append(
        pd.DataFrame({'总患者数': [col_count3], '有效频数(%)': [youxiao_pinshu_str3],
                      'Eta相关分析': [Eta_str]}), ignore_index=True)
    print("resultDf3:", resultDf3)
    # --------------------------- eta相关性分析end ------------------------------------
    # print("*" * 25)
    print("=" * 100, end="\n")

    # --------------------------- eta相关性分析end ------------------------------------
    # 观察变量在分组变量中的单因素分析
    # 做法：先将分组变量进行分组，统计分组数据内各个数据的频数，然后进行输出对应的指标即可
    # 分组变量
    fenzu_variable = "ABO_bloodType"
    # 观察变量
    guanca_variable = "age"

    # 对分组变量进行分组
    total_analysis = single_analysis(data, fenzu_variable, guanca_variable)
    print("单因素方差分析:", total_analysis)

    # -----------------------------聚类分析---------------------------------