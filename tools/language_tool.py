from opencc_purepy import OpenCC

class LanguageTool:
	@staticmethod
	def simplified_to_traditional(text: str) -> str:
		"""
		简繁转换
		:param text:简体
		:return:繁体
		"""
		cc = OpenCC("s2t")
		return cc.convert(text)