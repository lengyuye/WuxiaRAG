# 测试简繁转换
from opencc_purepy import OpenCC

# 简体 → 繁体
cc = OpenCC("s2t")
text = "春眠不觉晓，处处闻啼鸟。"
print(cc.convert(text))
# 春眠不覺曉，處處聞啼鳥。

# 简体 → 台湾正体（含地区词转换）
cc_tw = OpenCC("s2twp")
print(cc_tw.convert("计算机软件"))
# 電腦軟體

# 繁体 → 简体
cc_t2s = OpenCC("t2s")
print(cc_t2s.convert("這是一段繁體中文"))
# 这是一段繁体中文