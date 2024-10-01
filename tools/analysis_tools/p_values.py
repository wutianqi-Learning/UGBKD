from scipy import stats

# 前后样本数据
before = [76.21,
          77.2,
          76.56,
          76.27,
          76.53
          ]  # 前样本
after = [65.98,
         66.17,
         65.5,
         66.02,
         64.69
         ]  # 后样本

# 计算差值
differences = [after[i] - before[i] for i in range(len(before))]

# 计算差值的均值和标准差
mean_difference = sum(differences) / len(differences)
std_difference = (sum((x - mean_difference) ** 2 for x in differences) / (len(differences) - 1)) ** 0.5

# 计算t值
t = mean_difference / (std_difference / len(differences) ** 0.5)

# 计算自由度
df = len(differences) - 1

# 计算p值
p_value = stats.t.sf(abs(t), df) * 2  # 双尾检验

print("差值均值:", mean_difference)
print("差值标准差:", std_difference)
print("t值:", t)
print("p值:", p_value)
