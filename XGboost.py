import os
import pandas as pd
import numpy as np
extract_folder = 'BP_data/SVM_gaussian'

# Load data
extracted_files = os.listdir(extract_folder)
extracted_files.sort()
feature_files = [file for file in extracted_files if file.endswith('_features.csv')]
label_file = 'label.csv'

frames = []
for file in feature_files:
    file_path = os.path.join(extract_folder, file)
    df = pd.read_csv(file_path)
    frames.append(df)
features_df = pd.concat(frames, ignore_index=True)
label_df = pd.read_csv(os.path.join(extract_folder, label_file))

# Preprocessing
features_df.fillna(features_df.mean(), inplace=True)
features_df['P-T height avg'].fillna(0, inplace=True)
X = features_df
y = label_df[['Systolic Blood Pressures', 'Diastolic Blood Pressures']]


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create two models, one for SBP and one for DBP
xgb_model_sbp = make_pipeline(StandardScaler(), XGBRegressor(objective='reg:squarederror'))
xgb_model_dbp = make_pipeline(StandardScaler(), XGBRegressor(objective='reg:squarederror'))

# Train the models on their respective outputs
xgb_model_sbp.fit(X_train, y_train['Systolic Blood Pressures'])
xgb_model_dbp.fit(X_train, y_train['Diastolic Blood Pressures'])

# Predictions
y_pred_sbp = xgb_model_sbp.predict(X)
y_pred_dbp = xgb_model_dbp.predict(X)

# Evaluation for SBP and DBP
rmse_sbp = np.sqrt(mean_squared_error(y['Systolic Blood Pressures'], y_pred_sbp))
mae_sbp = mean_absolute_error(y['Systolic Blood Pressures'], y_pred_sbp)

rmse_dbp = np.sqrt(mean_squared_error(y['Diastolic Blood Pressures'], y_pred_dbp))
mae_dbp = mean_absolute_error(y['Diastolic Blood Pressures'], y_pred_dbp)

print("SBP RMSE:", '%.2f'%rmse_sbp)
# print("SBP MAE:", mae_sbp)
print("DBP RMSE:", '%.2f'%rmse_dbp)
# print("DBP MAE:", mae_dbp)

#################################################################################################
#################################################################################################
# Plotting
import matplotlib as mpl
def set_plot_style():
    # 设置全局参数
    # 设置图表的全局参数
    scale = 2
    mpl.rcParams['figure.figsize'] = (8, 6)  # 设置图表大小
    mpl.rcParams['axes.titlesize'] = 10*scale  # 设置轴标题大小
    mpl.rcParams['font.family'] = 'sans-serif'  # 设置全局字体家族为无衬线
    mpl.rcParams['font.sans-serif'] = ['Arial']  # 指定无衬线字体家族的具体字体
    mpl.rcParams['axes.labelsize'] = 10*scale  # 设置轴标签大小
    mpl.rcParams['xtick.labelsize'] = 8*scale  # 设置x轴刻度标签大小
    mpl.rcParams['ytick.labelsize'] = 8*scale  # 设置y轴刻度标签大小
    mpl.rcParams['legend.fontsize'] = 10*scale  # 设置图例字体大小

    mpl.rcParams['font.weight'] = 'bold'  # 设置字体加粗
    mpl.rcParams['axes.labelweight'] = 'bold'  # 加粗所有轴标题
    mpl.rcParams['axes.titleweight'] = 'normal'  # 设置轴标题字体加粗
    mpl.rcParams['figure.titleweight'] = 'normal'  # 图标题不加粗

    mpl.rcParams['lines.linewidth'] = 2  # 设置线条宽度
    mpl.rcParams['axes.grid'] = True  # 设置显示网格
    mpl.rcParams['grid.color'] = 'gray'  # 设置网格颜色
    mpl.rcParams['grid.linestyle'] = '--'  # 设置网格线条样式
    mpl.rcParams['grid.linewidth'] = 0.5  # 设置网格线条宽度
    # mpl.rcParams['figure.autolayout'] = True  # 自动调整布局
# 应用样式配置
set_plot_style()


# plt.rcParams['svg.fonttype'] = 'none'  # 确保文本不被转换为路径
# plt.savefig('figure/Actual vs Predicted Blood Pressure.svg', format='svg')



from sklearn.multioutput import MultiOutputRegressor

# Assuming other imports and data preprocessing are the same

# Create a multi-output model
xgb_multioutput_model = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(XGBRegressor(objective='reg:squarederror'))
)

# Train the model
xgb_multioutput_model.fit(X_train, y_train)

# Predictions
y_pred_multioutput = xgb_multioutput_model.predict(X)

# Evaluation
from sklearn.metrics import mean_squared_error

# 提取 SBP 和 DBP 的真实值和预测值
y_true_sbp = y['Systolic Blood Pressures']
y_pred_sbp = y_pred_multioutput[:, 0]
y_true_dbp = y['Diastolic Blood Pressures']
y_pred_dbp = y_pred_multioutput[:, 1]

# 分别计算 SBP 和 DBP 的 MSE
rmse_sbp = np.sqrt(mean_squared_error(y_true_sbp, y_pred_sbp))
rmse_dbp = np.sqrt(mean_squared_error(y_true_dbp, y_pred_dbp))

print("RMSE for Systolic Blood Pressure (SBP):", '%.2f'%rmse_sbp)
print("RMSE for Diastolic Blood Pressure (DBP):", '%.2f'%rmse_dbp)


#'%.2f'%

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Assuming X and y are already properly loaded and preprocessed
# X is the feature data, y is a DataFrame containing two columns: SBP and DBP

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multi-output model
xgb_multioutput_model = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(XGBRegressor(objective='reg:squarederror'))
)

# Train the model
xgb_multioutput_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_multioutput_model.predict(X)

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print("MSE:", mse)
print("MAE:", mae)

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Assuming X and y are already properly loaded and preprocessed
# X is the feature data, y is a DataFrame containing two columns: SBP and DBP

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multi-output model
xgb_multioutput_model = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(XGBRegressor(objective='reg:squarederror'))
)

# Train the model
xgb_multioutput_model.fit(X_train, y_train)

X_test = X
y_test = y

# Make predictions
y_pred = xgb_multioutput_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 计算 MAE 和 RMSE
mae_sbp = mean_absolute_error(y_test['Systolic Blood Pressures'], y_pred[:, 0])
mae_dbp = mean_absolute_error(y_test['Diastolic Blood Pressures'], y_pred[:, 1])

rmse_sbp = np.sqrt(mean_squared_error(y_test['Systolic Blood Pressures'], y_pred[:, 0]))
rmse_dbp = np.sqrt(mean_squared_error(y_test['Diastolic Blood Pressures'], y_pred[:, 1]))

# 计算误差的标准差
std_sbp = np.std(y_test['Systolic Blood Pressures'] - y_pred[:, 0])
std_dbp = np.std(y_test['Diastolic Blood Pressures'] - y_pred[:, 1])

# 打印结果
print(f"Systolic Blood Pressure MAE: {mae_sbp:.2f} ± {std_sbp:.2f} mmHg, RMSE: {rmse_sbp:.2f} mmHg ± {std_sbp:.2f} mmHg")
print(f"Diastolic Blood Pressure MAE: {mae_dbp:.2f} ± {std_dbp:.2f} mmHg, RMSE: {rmse_dbp:.2f} mmHg ± {std_dbp:.2f} mmHg")





# 1. Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
# 设置图像的方框为正方形
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('equal')
# 设置 X 轴和 Y 轴的范围
plt.xlim(50, 160)
plt.ylim(50, 160)
# 设置x轴刻度位置，间隔为20
plt.xticks(np.arange(40, 160 + 1, 20))
# 设置y轴刻度位置，间隔为20
plt.yticks(np.arange(40, 160 + 1, 20))

plt.scatter(y_test.iloc[:, 0], y_pred[:, 0], label='SBP', alpha=0.5, color = "#70a3c4")
plt.scatter(y_test.iloc[:, 1], y_pred[:, 1], label='DBP', alpha=0.5, color = "#df5b3f")
plt.plot([y.min().min(), y.max().max()], [y.min().min(), y.max().max()], 'k--', lw=2, color = '#666666')
plt.xlabel('Actual Blood Pressures')
plt.ylabel('Predicted Blood Pressures')
plt.title('Actual BP vs Predicted BP')
plt.legend()
plt.rcParams['svg.fonttype'] = 'none'  # 确保文本不被转换为路径
plt.savefig('figure/XGBoost_Actual BP vs Predicted BP.svg', format='svg')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_pred is a NumPy array and y_test is a DataFrame

# Calculate the errors
errors_sbp = y_test['Systolic Blood Pressures'] - y_pred[:, 0]  # SBP errors
errors_dbp = y_test['Diastolic Blood Pressures'] - y_pred[:, 1]  # DBP errors

# 2. Plot the error distribution using Kernel Density Estimation
plt.figure(figsize=(8, 6))
sns.kdeplot(errors_sbp, label='SBP Errors', shade=True)
sns.kdeplot(errors_dbp, label='DBP Errors', shade=True)
plt.xlabel('Error')
plt.ylabel('Density')
plt.title('Prediction Error Distribution')
plt.legend()
plt.xlim(-20, 20)
plt.rcParams['svg.fonttype'] = 'none'  # 确保文本不被转换为路径
plt.savefig('figure/XGBoost_Prediction Error Distribution.svg', format='svg')
plt.show()






# 2. Feature Importance Plot for XGBoost
import matplotlib.pyplot as plt
import numpy as np

# 获取特征重要性
# 这里我们使用第一个estimator作为代表，假设所有estimator的特征重要性相似
feature_importances = xgb_multioutput_model.named_steps['multioutputregressor'].estimators_[0].feature_importances_

# 创建特征名称与其重要性的字典
feature_importance_dict = dict(zip(X.columns, feature_importances))

# 过滤掉重要性低于特定阈值的特征
# 比如，设置阈值为0.01
threshold = 0.01
filtered_features = {feature: importance for feature, importance in feature_importance_dict.items() if importance >= threshold}

# 根据重要性排序
sorted_features = sorted(filtered_features.items(), key=lambda item: item[1], reverse=True)

# 解包排序后的特征和它们的重要性
sorted_feature_names, sorted_importances = zip(*sorted_features)

import matplotlib.pyplot as plt

# 假设 sorted_feature_names 和 sorted_importances 是已经排序的特征名称和它们的重要性

plt.figure(figsize=(8, 6))


bars = plt.barh(sorted_feature_names, sorted_importances, color = '#70a3c4')

# 在每个柱子旁边添加特征名称
for bar, feature_name in zip(bars, sorted_feature_names):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             ' {}'.format(feature_name),  # 在这里放置特征名称
             va='center', fontsize=10)

plt.gca().get_yaxis().set_visible(False)  # 隐藏 y 轴标签

plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Plot')
plt.gca().invert_yaxis()  # 翻转y轴，使得最重要的特征在上方
# 设置 X 轴和 Y 轴的范围
# plt.xlim(50, 160)
plt.rcParams['svg.fonttype'] = 'none'  # 确保文本不被转换为路径
plt.savefig('figure/Feature Importance Plot.svg', format='svg')
plt.show()




# 5. Validation Curve (Assuming you have collected the data)



# 1. Learning Curve (Assuming you have the training and validation scores)

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 假设 X 和 y 是您的特征和标签

# 分割数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化一个列表来存储每个 epoch 的 MSE
train_errors, val_errors = [], []

# 定义最大的树的数量（即 epoch 的数量）
n_estimators = 100

for n in range(1, n_estimators + 1):
    # 更新模型，仅增加一个树
    model = MultiOutputRegressor(XGBRegressor(n_estimators=n, objective='reg:squarederror'))
    model.fit(X_train, y_train)

    # 在训练集和验证集上进行预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 计算并记录 MSE
    train_errors.append(mean_squared_error(y_train, y_train_pred, multioutput='raw_values'))
    val_errors.append(mean_squared_error(y_val, y_val_pred, multioutput='raw_values'))

# 计算均值以获取单一的误差度量
train_errors_mean = np.mean(train_errors, axis=1)
val_errors_mean = np.mean(val_errors, axis=1)

# # 绘制学习曲线
# plt.figure(figsize=(8, 6))
# plt.plot(np.sqrt(train_errors_mean), label='Train')
# plt.plot(np.sqrt(val_errors_mean), label='Validation')
# plt.xlabel('Number of Trees')
# plt.ylabel('RMSE')
# plt.title('Learning Curve')
# plt.legend()
# plt.rcParams['svg.fonttype'] = 'none'  # 确保文本不被转换为路径
# plt.savefig('figure/XGBoost_Learning Curve.svg', format='svg')
# plt.show()



# # 5. Blossom Plot (桑葚图)
# import plotly.graph_objects as go
#
# # 假设 y_test 和 y_pred 已经存在，并且是 DataFrame 和 NumPy 数组
#
# # 提取数据
# actual_sbp = y_test['Systolic Blood Pressures']
# predicted_sbp = y_pred[:, 0]
# actual_dbp = y_test['Diastolic Blood Pressures']
# predicted_dbp = y_pred[:, 1]
#
# # 桑葚图的节点和链接
# labels = ['Actual SBP', 'Predicted SBP', 'Actual DBP', 'Predicted DBP']
# source = [0, 0, 1, 2]
# target = [1, 2, 3, 3]
# value = [actual_sbp.mean(), predicted_sbp.mean(), actual_dbp.mean(), predicted_dbp.mean()]
#
# # 创建桑葚图
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=labels,
#         color=["#e6211a", "#4a7ab5", "#ec9e52", "#76b79b"]
#     ),
#     link=dict(
#         source=source,  # 起点节点
#         target=target,  # 终点节点
#         value=value,    # 流量值
#         color=["rgb(252, 153, 153)", "rgb(202, 215, 238)", "rgb(247, 209, 119)", "rgb(198, 222, 181)"]
#     )
# )])
#
# fig.update_layout(title_text="Sankey Diagram of Blood Pressure Predictions", font_size=40)
# fig.show()







# 假设 y_test 和 y_pred 已经存在，并且是 DataFrame 和 NumPy 数组

# 提取数据
# actual_sbp = y_test['Systolic Blood Pressures']
# predicted_sbp = y_pred[:, 0]
# actual_dbp = y_test['Diastolic Blood Pressures']
# predicted_dbp = y_pred[:, 1]
#
# # 桑葚图的节点和链接
# labels = ['Actual SBP', 'Predicted SBP', 'Actual DBP', 'Predicted DBP']
# source = [0, 0, 1, 2]
# target = [1, 2, 3, 3]
# value = [actual_sbp.mean(), predicted_sbp.mean(), actual_dbp.mean(), predicted_dbp.mean()]
#
# # 创建桑葚图
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=labels,
#         color=["#e6211a", "#4a7ab5", "#ec9e52", "#76b79b"]
#     ),
#     link=dict(
#         source=source,  # 起点节点
#         target=target,  # 终点节点
#         value=value,    # 流量值
#         color=["rgba(252, 153, 153)", "rgba(202, 215, 238)", "rgba(247, 209, 119)", "rgba(198, 222, 181)"]
#     )
# )])
#
# fig.update_layout(title_text="Sankey Diagram of Blood Pressure Predictions", font_size=10)
#
# # 保存为 SVG 文件
# fig.write_image('figure/Sankey_Diagram.svg')




