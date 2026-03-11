# -*- coding: utf-8 -*-
"""
===================================
股票自選股智能分析系統 - AI 分析層
===================================
職責：
1. 封裝 LLM 調用邏輯（透過 LiteLLM 統一調用 Gemini/OpenAI 等）
2. 結合技術面與消息面生成繁體分析報告
3. 解析 LLM 響應為結構化 AnalysisResult
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config, get_api_keys_for_model, extra_litellm_params

logger = logging.getLogger(__name__)

# 股票名稱映射（繁體化與精簡）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '貴州茅台', '000001': '平安銀行', '300750': '寧德時代', '002594': '比亞迪',
    '600036': '招商銀行', '601318': '中國平安', '000858': '五糧液', '600276': '恆瑞醫藥',

    # === 美股 ===
    'AAPL': '蘋果', 'TSLA': '特斯拉', 'MSFT': '微軟', 'GOOGL': '谷歌A',
    'NVDA': '輝達', 'META': 'Meta', 'AMZN': '亞馬遜', 'AMD': 'AMD',
    'COIN': 'Coinbase', 'MSTR': '微策略',

    # === 港股 ===
    '00700': '騰訊控股', '03690': '美團', '01810': '小米集團', '09988': '阿里巴巴'
}

def get_stock_name_multi_source(stock_code: str, context: Optional[Dict] = None, data_manager = None) -> str:
    """多來源獲取股票中文名稱"""
    if context:
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'): return name
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']
    return STOCK_NAME_MAP.get(stock_code, f'股票{stock_code}')

@dataclass
class AnalysisResult:
    """AI 分析結果數據類 - 繁體版"""
    code: str
    name: str
    sentiment_score: int  # 綜合評分 0-100
    trend_prediction: str # 趨勢預測
    operation_advice: str # 操作建議
    decision_type: str = "hold"
    confidence_level: str = "中"
    dashboard: Optional[Dict[str, Any]] = None 
    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""
    market_snapshot: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def get_emoji(self) -> str:
        emoji_map = {'買入': '🟢', '加倉': '🟢', '持有': '🟡', '觀望': '⚪', '減倉': '🟠', '賣出': '🔴'}
        advice = self.operation_advice or ''
        for k, v in emoji_map.items():
            if k in advice: return v
        return '⚪'

class GeminiAnalyzer:
    """Gemini AI 分析器 (精簡繁體版)"""

    SYSTEM_PROMPT = """你是一位專業投資分析師，請為股票生成【決策儀表板】報告。

## 交易原則：
1. 嚴進策略：不追高，乖離率 > 5% 觀望。
2. 趨勢交易：僅做 MA5>MA10>MA20 多頭排列股票。
3. 效率優先：關注籌碼集中度與量能配合。
4. 買點偏好：偏好縮量回踩均線支撐。

## 輸出格式 (嚴格 JSON)：
{
    "stock_name": "正確全稱",
    "sentiment_score": 0-100,
    "trend_prediction": "看多/震盪/看空",
    "operation_advice": "買入/持有/觀望/賣出",
    "decision_type": "buy/hold/sell",
    "dashboard": {
        "core_conclusion": { "one_sentence": "30字內建議", "signal_type": "符號+訊號", "position_advice": {"no_position": "指引", "has_position": "指引"} },
        "data_perspective": { "trend_status": {"ma_alignment": "狀態"}, "price_position": {"bias_ma5": "數值", "bias_status": "安全/危險", "support": "價位", "resistance": "價位"} },
        "battle_plan": { "sniper_points": {"ideal_buy": "價位", "stop_loss": "價位", "target": "價位"}, "action_checklist": ["✅/❌ 檢查項"] }
    },
    "analysis_summary": "100字摘要",
    "risk_warning": "核心風險"
}"""

    def __init__(self, api_key: Optional[str] = None):
        self._router = None
        self._litellm_available = False
        self._init_litellm()

    def _init_litellm(self) -> None:
        config = get_config()
        if not config.litellm_model: return
        self._litellm_available = True
        if bool(config.llm_model_list):
            self._router = Router(model_list=config.llm_model_list, routing_strategy="simple-shuffle", num_retries=2)
            logger.info(f"Analyzer: 路由模式已啟動 ({config.litellm_model})")

    def is_available(self) -> bool:
        return self._litellm_available

    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None) -> AnalysisResult:
        code = context.get('code', 'Unknown')
        name = get_stock_name_multi_source(code, context)
        
        if not self.is_available():
            return AnalysisResult(code=code, name=name, sentiment_score=50, trend_prediction='未知', operation_advice='觀望', success=False, error_message='未配置 API Key')

        try:
            prompt = self._format_prompt(context, name, news_context)
            logger.info(f"AI 分析啟動: {name}({code}) | 模型: {get_config().litellm_model}")
            
            res_text, model_used = self._call_litellm(prompt, {"temperature": get_config().gemini_temperature})
            result = self._parse_response(res_text, code, name)
            result.raw_response, result.model_used, result.search_performed = res_text, model_used, bool(news_context)
            return result
        except Exception as e:
            logger.error(f"分析失敗 {name}: {e}")
            return AnalysisResult(code=code, name=name, sentiment_score=50, trend_prediction='錯誤', operation_advice='觀望', success=False, error_message=str(e))

    def _format_prompt(self, context: Dict[str, Any], name: str, news_context: Optional[str] = None) -> str:
        today = context.get('today', {})
        rt = context.get('realtime', {})
        trend = context.get('trend_analysis', {})
        
        return f"""# 股票分析請求: {name}({context.get('code')})
## 技術指標:
- 現價: {today.get('close')} | 漲跌: {today.get('pct_chg')}%
- 均線: MA5({today.get('ma5')}), MA10({today.get('ma10')}), MA20({today.get('ma20')})
- 乖離率: {trend.get('bias_ma5', 0):.2f}% | 形態: {trend.get('ma_alignment')}
- 量能變化: {context.get('volume_change_ratio')}倍 | 換手率: {rt.get('turnover_rate')}%

## 輿情摘要:
{news_context if news_context else "無近期新聞"}

請依據上述數據產出繁體中文【決策儀表板】JSON。"""

    def _call_litellm(self, prompt: str, gen_config: dict) -> Tuple[str, str]:
        config = get_config()
        model = config.litellm_model
        kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "temperature": gen_config.get("temperature", 0.7),
            "max_tokens": 4096
        }
        extra = get_thinking_extra_body(model)
        if extra: kwargs["extra_body"] = extra

        res = self._router.completion(**kwargs) if self._router else litellm.completion(**kwargs)
        return res.choices[0].message.content, model

    def _parse_response(self, text: str, code: str, name: str) -> AnalysisResult:
        try:
            json_str = repair_json(text[text.find('{'):text.rfind('}')+1])
            data = json.loads(json_str)
            return AnalysisResult(
                code=code, name=data.get('stock_name', name),
                sentiment_score=int(data.get('sentiment_score', 50)),
                trend_prediction=data.get('trend_prediction', '震盪'),
                operation_advice=data.get('operation_advice', '持有'),
                dashboard=data.get('dashboard'),
                analysis_summary=data.get('analysis_summary', ''),
                risk_warning=data.get('risk_warning', ''),
                success=True
            )
        except:
            return self._parse_text_fallback(text, code, name)

    def _parse_text_fallback(self, text: str, code: str, name: str) -> AnalysisResult:
        logger.warning("JSON 解析失敗，啟用文字提取模式")
        return AnalysisResult(code=code, name=name, sentiment_score=50, trend_prediction='解析中', operation_advice='觀望', analysis_summary=text[:500], success=True)

# 輔助格式化方法 (省略部分與原代碼相同的 format_volume 等)
