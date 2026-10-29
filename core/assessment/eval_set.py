# -*- coding: utf-8 -*-
"""评估集：从制度原文反推的问题 + 标准答案 + 依据条款。

为什么不用手写题目：
    手写题目很容易问出制度里根本没有的内容，导致评估指标反映的是"题库质量"
    而不是"系统能力"。这里每一条问题都从真实条款反推——
    先锁定条款，再据条款正文写问法，ground_truth 直接取自条款原文的核心句子。
    这样 context_recall 才有意义：系统没召回到该条款，就是真没召回。

按业务条线分层抽样：
    个人信贷 / 反洗钱 / 运营柜面 / 跨条线时效 / 纯拒答 五个层，
    每层的题量与线上问题分布大致同比例（信贷问题最多，拒答题少但必须覆盖）。
    分层的好处是能看出"哪个条线弱"——总分相同，分层结果可能完全不同。

字段说明：
    question      用户会怎么问（贴近柜面/客户经理的真实说法，不是念条款标题）
    ground_truth  标准答案（取自条款原文，可逐字核对）
    clause_key    依据条款，格式 文号#条款号
    doc_no        所属制度文号
    line          业务条线（分层抽样的层标签）
    type          numeric 数字查询 / factual 事实查询 / procedural 流程查询
                  / temporal 时效查询 / refuse 应拒答
    expect_refuse 是否期望系统拒答（问制度范围外的内容）
"""
from __future__ import annotations

# 条线常量：与 config.ini 的 business_lines 对齐，另加两个评估专用层
LINE_CREDIT = "个人信贷"
LINE_AML = "反洗钱"
LINE_COUNTER = "运营柜面"
LINE_CROSS = "跨条线时效"
LINE_REFUSE = "纯拒答"

EVAL_SET = [
    # ==================== 个人信贷（8 条） ====================
    # 数字类：制度问答最容易答错的就是数字
    {
        "question": "个人消费贷款最多能贷多少钱？",
        "ground_truth": "信用类消费贷款单户最高不超过人民币50万元。",
        "clause_key": "恒信银发〔2024〕27号#第七条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "抵押类消费贷款最高能贷多少？",
        "ground_truth": "抵押类消费贷款单户最高不超过人民币500万元。",
        "clause_key": "恒信银发〔2024〕27号#第七条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "个人经营贷款单户最多能贷多少？",
        "ground_truth": "个人经营贷款单户最高不超过人民币1000万元。",
        "clause_key": "恒信银发〔2024〕27号#第七条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "多少额度以内的贷款分行可以直接批？",
        "ground_truth": "单笔金额50万元（含）以下，由分行有权审批人审批。",
        "clause_key": "恒信银发〔2024〕27号#第九条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "一笔三百万的个人贷款要报到哪一级审批？",
        "ground_truth": "单笔金额200万元以上，报总行零售信贷审批中心审批。",
        "clause_key": "恒信银发〔2024〕27号#第九条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "numeric",
        "expect_refuse": False,
    },
    # 事实类
    {
        "question": "贷款用途有哪些是不能做的？",
        "ground_truth": "个人贷款用途应当符合法律法规规定和国家有关政策，不得用于下列用途：（一）购房首付款、房地产开发相关支出；（二）股票、债券、期货、金融衍生品和资产管理产品投资；（三）股本权益性投资。",
        "clause_key": "恒信银发〔2024〕27号#第五条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "factual",
        "expect_refuse": False,
    },
    {
        "question": "借款人月债务收入比不能超过多少？",
        "ground_truth": "借款人月债务收入比（DTI）原则上不超过55%。",
        "clause_key": "恒信银发〔2024〕27号#第六条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "numeric",
        "expect_refuse": False,
    },
    # 流程类
    {
        "question": "贷款逾期以后多久要开始催收？",
        "ground_truth": "个人贷款逾期后，经办行应当在逾期3个工作日内启动催收。",
        "clause_key": "恒信银发〔2024〕27号#第十五条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CREDIT,
        "type": "procedural",
        "expect_refuse": False,
    },

    # ==================== 反洗钱（8 条） ====================
    # 事实类
    {
        "question": "客户哪些信息属于受益所有人？",
        "ground_truth": "受益所有人是指最终拥有或者实际控制客户，或者享有客户最终收益的自然人。",
        "clause_key": "恒信银发〔2025〕3号#第五条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "factual",
        "expect_refuse": False,
    },
    {
        "question": "受益所有人持股多少比例以上要被认定？",
        "ground_truth": "直接或间接拥有客户25%以上股权或表决权的自然人应当认定为受益所有人。",
        "clause_key": "恒信银发〔2025〕3号#第五条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "客户风险等级一共分几级？",
        "ground_truth": "客户洗钱风险等级分为低风险、中低风险、中风险、中高风险、高风险五级。",
        "clause_key": "恒信银发〔2025〕3号#第七条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "factual",
        "expect_refuse": False,
    },
    # 数字类
    {
        "question": "新客户开户后多久要完成风险等级评定？",
        "ground_truth": "新建立业务关系的客户，应当在建立业务关系后10个工作日内完成风险等级评定。",
        "clause_key": "恒信银发〔2025〕3号#第七条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "哪些情况要启动可疑交易分析？",
        "ground_truth": "出现下列情形之一的应当启动可疑交易分析：（一）单日现金存取累计超过人民币20万元；（二）客户账户资金快进快出，且无合理商业理由。",
        "clause_key": "恒信银发〔2025〕3号#第九条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "factual",
        "expect_refuse": False,
    },
    # 流程类
    {
        "question": "高风险客户多久要重新审核一次身份信息？",
        "ground_truth": "对高风险客户应当每半年至少重新审核一次客户身份信息和受益所有人信息。",
        "clause_key": "恒信银发〔2025〕3号#第八条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "procedural",
        "expect_refuse": False,
    },
    {
        "question": "客户身份资料和交易记录要保存多久？",
        "ground_truth": "客户身份资料和交易记录应当自业务关系结束当年或者一次性交易记账当年计起至少保存5年。",
        "clause_key": "恒信银发〔2025〕3号#第十条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "什么情况下可以简化尽职调查？",
        "ground_truth": "对同时符合下列条件的低风险客户，可以采取简化尽职调查措施：（一）客户风险等级评定为低风险；（二）客户交易为工资代发、社保发放、公用事业缴费等业务；（三）单笔交易金额不超过人民币1万元。",
        "clause_key": "恒信银发〔2025〕3号#第六条",
        "doc_no": "恒信银发〔2025〕3号",
        "line": LINE_AML,
        "type": "procedural",
        "expect_refuse": False,
    },

    # ==================== 运营柜面（8 条） ====================
    # 流程类
    {
        "question": "开户的时候柜员要核验什么？",
        "ground_truth": "开户时应当：（一）核验身份证件真实性和有效性，通过联网核查系统比对；（二）采集并留存客户影像、身份证件影像；（三）登记客户职业、联系方式、住所地址等基本信息。",
        "clause_key": "恒信银运〔2024〕58号#第三条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "procedural",
        "expect_refuse": False,
    },
    {
        "question": "一个人能在我们行开几个Ⅰ类户？",
        "ground_truth": "原则上每人每家商业银行只能开立一个Ⅰ类户。已开立Ⅰ类户的，再开户时应当开立Ⅱ类户或Ⅲ类户。",
        "clause_key": "恒信银运〔2024〕58号#第三条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "factual",
        "expect_refuse": False,
    },
    # 数字类
    {
        "question": "单笔多少现金存取要核对身份证并登记来源用途？",
        "ground_truth": "办理单笔或当日累计人民币5万元以上现金存取业务的，应当核对客户有效身份证件，并登记资金来源或用途。",
        "clause_key": "恒信银运〔2024〕58号#第六条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "手机银行短信验证方式转账限额是多少？",
        "ground_truth": "短信验证方式：单笔5万元，日累计5万元。",
        "clause_key": "恒信银运〔2024〕58号#第九条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "柜面转账超过多少要网点负责人授权？",
        "ground_truth": "柜面转账单笔不设上限，超过人民币50万元的应当由网点负责人授权。",
        "clause_key": "恒信银运〔2024〕58号#第九条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "临时挂失有效期几天？",
        "ground_truth": "临时挂失有效期5天。",
        "clause_key": "恒信银运〔2024〕58号#第十二条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "numeric",
        "expect_refuse": False,
    },
    {
        "question": "正式挂失以后多久能补办新卡？",
        "ground_truth": "正式挂失后7天可以补办新卡或支取。",
        "clause_key": "恒信银运〔2024〕58号#第十二条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "procedural",
        "expect_refuse": False,
    },
    {
        "question": "老年人给陌生账户大额转账柜面要怎么处理？",
        "ground_truth": "柜面发现老年客户大额转账至陌生账户的，应当进行风险提示并核实；经劝阻无效的，应当报告网点负责人并视情况报警。",
        "clause_key": "恒信银运〔2024〕58号#第十一条",
        "doc_no": "恒信银运〔2024〕58号",
        "line": LINE_COUNTER,
        "type": "procedural",
        "expect_refuse": False,
    },

    # ==================== 跨条线 + 时效（4 条） ====================
    # 这层专测"新旧办法并存"这个最容易出事的地方
    {
        "question": "个人消费贷款额度上限现在执行哪个口径？",
        "ground_truth": "个人消费贷款单户额度上限按照产品类别分别执行：（一）信用类消费贷款单户最高不超过人民币50万元；（二）抵押类消费贷款单户最高不超过人民币500万元。",
        "clause_key": "恒信银发〔2024〕27号#第七条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CROSS,
        "type": "temporal",
        "expect_refuse": False,
    },
    {
        "question": "旧的那个暂行办法还算数吗？",
        "ground_truth": "原《恒信银行个人贷款管理暂行办法》（恒信银发〔2021〕9号）自本办法施行之日起废止。",
        "clause_key": "恒信银发〔2024〕27号#第十六条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CROSS,
        "type": "temporal",
        "expect_refuse": False,
    },
    {
        "question": "消费贷款期限最长几年？",
        "ground_truth": "个人消费贷款期限最长不超过5年；个人经营贷款期限最长不超过10年。",
        "clause_key": "恒信银发〔2024〕27号#第八条",
        "doc_no": "恒信银发〔2024〕27号",
        "line": LINE_CROSS,
        "type": "temporal",
        "expect_refuse": False,
    },
    {
        "question": "2021年那个办法规定的消费贷额度是多少？",
        "ground_truth": "《恒信银行个人贷款管理暂行办法》（恒信银发〔2021〕9号）第六条规定个人消费贷款单户额度最高不超过人民币30万元，但该办法已于2024年7月1日废止，现行应执行50万元口径。",
        "clause_key": "恒信银发〔2021〕9号#第六条",
        "doc_no": "恒信银发〔2021〕9号",
        "line": LINE_CROSS,
        "type": "temporal",
        "expect_refuse": False,
    },

    # ==================== 纯拒答（4 条） ====================
    # 制度里没有的内容，答了就是幻觉
    {
        "question": "恒信银行2025年的净利润是多少？",
        "ground_truth": "（制度库中无此信息，应拒答或说明无法回答）",
        "clause_key": "",
        "doc_no": "",
        "line": LINE_REFUSE,
        "type": "refuse",
        "expect_refuse": True,
    },
    {
        "question": "央行今天的LPR报价是多少？",
        "ground_truth": "（制度库中无此信息，应拒答或说明无法回答）",
        "clause_key": "",
        "doc_no": "",
        "line": LINE_REFUSE,
        "type": "refuse",
        "expect_refuse": True,
    },
    {
        "question": "帮我写一份离职申请书。",
        "ground_truth": "（与制度问答无关，应拒答）",
        "clause_key": "",
        "doc_no": "",
        "line": LINE_REFUSE,
        "type": "refuse",
        "expect_refuse": True,
    },
    {
        "question": "我这张信用卡的具体年费是多少？",
        "ground_truth": "（制度库中无信用卡年费标准，属个案业务信息，应拒答并引导查询个人账单或联系客服）",
        "clause_key": "",
        "doc_no": "",
        "line": LINE_REFUSE,
        "type": "refuse",
        "expect_refuse": True,
    },
]


def stats() -> dict:
    """按类型统计题量，评估报告里用来交代覆盖面。"""
    counter: dict[str, int] = {}
    for item in EVAL_SET:
        counter[item["type"]] = counter.get(item["type"], 0) + 1
    return counter


def stats_by_line() -> dict:
    """按业务条线统计题量，用来证明"分层抽样"不是一句空话。"""
    counter: dict[str, int] = {}
    for item in EVAL_SET:
        line = item.get("line") or "未标注"
        counter[line] = counter.get(line, 0) + 1
    return counter


def retrieval_only() -> list[dict]:
    """只保留有依据条款的题目，供纯检索层评估使用。

    拒答题没有标准条款，算 context_recall 会污染分母，
    它们只在端到端的拒答准确率里统计。
    """
    return [item for item in EVAL_SET if item["clause_key"]]


if __name__ == "__main__":
    total = len(EVAL_SET)
    by_type = stats()
    by_line = stats_by_line()
    print(f"评估集共 {total} 条")
    print("-" * 46)
    print("按类型：")
    for key, value in by_type.items():
        print(f"  {key:12s} {value:3d} 条  {value / total:5.1%}")
    print("按业务条线：")
    for key, value in by_line.items():
        print(f"  {key:10s} {value:3d} 条  {value / total:5.1%}")
    print("-" * 46)
    print(f"可参与检索评估（有依据条款）：{len(retrieval_only())} 条")
    print(f"应拒答：{sum(1 for i in EVAL_SET if i['expect_refuse'])} 条")
