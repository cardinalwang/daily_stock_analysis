# -*- coding: utf-8 -*-
"""
===================================
A股自選股智能分析系統 - AI分析層
===================================

職責：
1. 封裝 LLM 調用邏輯（透過 LiteLLM 統一調用 Gemini/Anthropic/OpenAI 等）
2. 結合技術面和消息面生成分析報告
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


# 股票名稱映射（常見股票）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '貴州茅台',
    '000001': '平安銀行',
    '300750': '寧德時代',
    '002594': '比亞迪',
    '600036': '招商銀行',
    '601318': '中國平安',
    '000858': '五糧液',
    '600276': '恆瑞醫藥',
    '601012': '隆基綠能',
    '002475': '立訊精密',
    '300059': '東方財富',
    '002415': '海康威視',
    '600900': '長江電力',
    '601166': '興業銀行',
    '600028': '中國石化',

    # === 美股 ===
    'AAPL': '蘋果',
    'TSLA': '特斯拉',
    'MSFT': '微軟',
    'GOOGL': '谷歌A',
    'GOOG': '谷歌C',
    'AMZN': '亞馬遜',
    'NVDA': '輝達',
    'META': 'Meta',
    'AMD': 'AMD',
    'INTC': '英特爾',
    'BABA': '阿里巴巴',
    'PDD': '拼多多',
    'JD': '京東',
    'BIDU': '百度',
    'NIO': '蔚來',
    'XPEV': '小鵬汽車',
    'LI': '理想汽車',
    'COIN': 'Coinbase',
    'MSTR': 'MicroStrategy',

    # === 港股 (5位數字) ===
    '00700': '騰訊控股',
    '03690': '美團',
    '01810': '小米集團',
    '09988': '阿里巴巴',
    '09618': '京東集團',
    '09888': '百度集團',
    '01024': '快手',
    '00981': '中芯國際',
    '02015': '理想汽車',
    '09868': '小鵬汽車',
    '00005': '匯豐控股',
    '01299': '友邦保險',
    '00941': '中國移動',
    '00883': '中國海洋石油',
}


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """多來源獲取股票中文名稱"""
    # 1. 從上下文獲取（實時行情數據）
    if context:
        # 優先從 stock_name 字段獲取
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name

        # 其次從 realtime 數據獲取
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. 從靜態映射表獲取
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. 從數據源獲取
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"無法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                # 更新緩存
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"從數據源獲取股票名稱失敗: {e}")

    # 4. 返回默認名稱
    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """AI 分析結果數據類 - 決策儀表板版"""
    code: str
    name: str

    # ========== 核心指標 ==========
    sentiment_score: int  # 綜合評分 0-100 (>70強烈看多, >60看多, 40-60震盪, <40看空)
    trend_prediction: str  # 趨勢預測：強烈看多/看多/震盪/看空/強烈看空
    operation_advice: str  # 操作建議：買入/加倉/持有/減倉/賣出/觀望
    decision_type: str = "hold"  # 決策類型：buy/hold/sell（用於統計）
    confidence_level: str = "中"  # 置信度：高/中/低

    # ========== 決策儀表板 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的決策儀表板數據

    # ========== 走勢分析 ==========
    trend_analysis: str = ""  # 走勢形態分析（支撐位、壓力位、趨勢線等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2週）

    # ========== 技術面分析 ==========
    technical_analysis: str = ""  # 技術指標綜合分析
    ma_analysis: str = ""  # 均線分析（多頭/空頭排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/縮量，主力動向等）
    pattern_analysis: str = ""  # K線形態分析

    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面綜合分析
    sector_position: str = ""  # 板塊地位和行業趨勢
    company_highlights: str = ""  # 公司亮點/風險點

    # ========== 情緒面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新聞/公告摘要
    market_sentiment: str = ""  # 市場情緒分析
    hot_topics: str = ""  # 相關熱點話題

    # ========== 綜合分析 ==========
    analysis_summary: str = ""  # 綜合分析摘要
    key_points: str = ""  # 核心看點（3-5個要點）
    risk_warning: str = ""  # 風險提示
    buy_reason: str = ""  # 買入/賣出理由

    # ========== 元數據 ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # 當日行情快照（展示用）
    raw_response: Optional[str] = None  # 原始響應（調試用）
    search_performed: bool = False  # 是否執行了聯網搜索
    data_sources: str = ""  # 數據來源說明
    success: bool = True
    error_message: Optional[str] = None

    # ========== 價格數據（分析時快照）==========
    current_price: Optional[float] = None  # 分析時的股價
    change_pct: Optional[float] = None     # 分析時的漲跌幅(%)

    # ========== 模型標記（Issue #528）==========
    model_used: Optional[str] = None  # 分析使用的 LLM 模型（完整名，如 gemini/gemini-2.0-flash）

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
            'model_used': self.model_used,
        }

    def get_core_conclusion(self) -> str:
        """獲取核心結論（一句話）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """獲取持倉建議"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """獲取狙擊點位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """獲取檢查清單"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """獲取風險警報"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """根據操作建議返回對應 emoji"""
        emoji_map = {
            '買入': '🟢', '加倉': '🟢', '強烈買入': '💚', '持有': '🟡',
            '觀望': '⚪', '減倉': '🟠', '賣出': '🔴', '強烈賣出': '❌',
        }
        advice = self.operation_advice or ''
        if advice in emoji_map:
            return emoji_map[advice]
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        score = self.sentiment_score
        if score >= 80: return '💚'
        elif score >= 65: return '🟢'
        elif score >= 55: return '🟡'
        elif score >= 45: return '⚪'
        elif score >= 35: return '🟠'
        else: return '🔴'

    def get_confidence_stars(self) -> str:
        """返回置信度星級"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """Gemini AI 分析器"""

    # 確保字串常數不會因為排版錯誤引發 SyntaxError
    SYSTEM_PROMPT = (
        "你是一位專注於趨勢交易的 A 股投資分析師，負責生成專業的【決策儀表板】分析報告。\n\n"
        "## 重要指令：必須使用【繁體中文】(Taiwan Chinese) 回答所有內容，嚴禁使用簡體字或英文術語。\n\n"
        "## 核心交易理念（必須嚴格遵守）\n\n"
        "### 1. 嚴進策略（不追高）\n"
        "- **絕對不追高**：當股價偏離 MA5 超過 5% 時，堅決不買入\n"
        "- **乖離率公式**：(現價 - MA5) / MA5 × 100%\n"
        "- 乖離率 < 2%：最佳買點區間\n"
        "- 乖離率 2-5%：可小倉介入\n"
        "- 乖離率 > 5%：嚴禁追高！直接判定為\"觀望\"\n\n"
        "### 2. 趨勢交易（順勢而為）\n"
        "- **多頭排列必須條件**：MA5 > MA10 > MA20\n"
        "- 只做多頭排列的股票，空頭排列堅決不碰\n"
        "- 均線發散上行優於均線粘合\n"
        "- 趨勢強度判斷：看均線間距是否在擴大\n\n"
        "### 3. 效率優先（籌碼結構）\n"
        "- 關注籌碼集中度：90%集中度 < 15% 表示籌碼集中\n"
        "- 獲利比例分析：70-90% 獲利盤時需警惕獲利回吐\n"
        "- 平均成本與現價關係：現價高於平均成本 5-15% 為健康\n\n"
        "### 4. 買點偏好（回踩支撐）\n"
        "- **最佳買點**：縮量回踩 MA5 獲得支撐\n"
        "- **次優買點**：回踩 MA10 獲得支撐\n"
        "- **觀望情況**：跌破 MA20 時觀望\n\n"
        "### 5. 風險排查重點\n"
        "- 減持公告（股東、高管減持）\n"
        "- 業績預虧/大幅下滑\n"
        "- 監管處罰/立案調查\n"
        "- 行業政策利空\n"
        "- 大額解禁\n\n"
        "### 6. 估值關注（PE/PB）\n"
        "- 分析時請關注本益比（PE）是否合理\n"
        "- PE 明顯偏高時（如遠超行業平均或歷史均值），需在風險點中說明\n"
        "- 高成長股可適當容忍較高 PE，但需有業績支撐\n\n"
        "### 7. 強勢趨勢股放寬\n"
        "- 強勢趨勢股（多頭排列且趨勢強度高、量能配合）可適當放寬乖離率要求\n"
        "- 此類股票可輕倉追蹤，但仍需設置止損，不盲目追高\n\n"
        "## 輸出格式：決策儀表板 JSON\n\n"
        "請嚴格按照以下 JSON 格式輸出，這是一個完整的【決策儀表板】：\n\n"
        "```json\n"
        "{\n"
        "    \"stock_name\": \"股票中文名稱\",\n"
        "    \"sentiment_score\": 50,\n"
        "    \"trend_prediction\": \"強烈看多/看多/震盪/看空/強烈看空\",\n"
        "    \"operation_advice\": \"買入/加倉/持有/減倉/賣出/觀望\",\n"
        "    \"decision_type\": \"buy/hold/sell\",\n"
        "    \"confidence_level\": \"高/中/低\",\n"
        "    \"dashboard\": {\n"
        "        \"core_conclusion\": {\n"
        "            \"one_sentence\": \"一句話核心結論（30字以內，直接告訴用戶做什麼）\",\n"
        "            \"signal_type\": \"🟢買入信號/🟡持有觀望/🔴賣出信號/⚠️風險警告\",\n"
        "            \"time_sensitivity\": \"立即行動/今日內/本週內/不急\",\n"
        "            \"position_advice\": {\n"
        "                \"no_position\": \"空倉者建議：具體操作指引\",\n"
        "                \"has_position\": \"持倉者建議：具體操作指引\"\n"
        "            }\n"
        "        },\n"
        "        \"data_perspective\": {\n"
        "            \"trend_status\": {\n"
        "                \"ma_alignment\": \"均線排列狀態描述\",\n"
        "                \"is_bullish\": true,\n"
        "                \"trend_score\": 80\n"
        "            },\n"
        "            \"price_position\": {\n"
        "                \"current_price\": 10.0,\n"
        "                \"ma5\": 9.5,\n"
        "                \"ma10\": 9.0,\n"
        "                \"ma20\": 8.5,\n"
        "                \"bias_ma5\": 5.2,\n"
        "                \"bias_status\": \"安全/警戒/危險\",\n"
        "                \"support_level\": 9.0,\n"
        "                \"resistance_level\": 11.0\n"
        "            },\n"
        "            \"volume_analysis\": {\n"
        "                \"volume_ratio\": 1.5,\n"
        "                \"volume_status\": \"放量/縮量/平量\",\n"
        "                \"turnover_rate\": 5.0,\n"
        "                \"volume_meaning\": \"量能含義解讀（如：縮量回調表示拋壓減輕）\"\n"
        "            },\n"
        "            \"chip_structure\": {\n"
        "                \"profit_ratio\": 0.8,\n"
        "                \"avg_cost\": 8.0,\n"
        "                \"concentration\": 0.1,\n"
        "                \"chip_health\": \"健康/一般/警惕\"\n"
        "            }\n"
        "        },\n"
        "        \"intelligence\": {\n"
        "            \"latest_news\": \"【最新消息】近期重要新聞摘要\",\n"
        "            \"risk_alerts\": [\"風險點1：具體描述\", \"風險點2：具體描述\"],\n"
        "            \"positive_catalysts\": [\"利好1：具體描述\", \"利好2：具體描述\"],\n"
        "            \"earnings_outlook\": \"業績預期分析（基於年報預告、業績快報等）\",\n"
        "            \"sentiment_summary\": \"輿情情緒一句話總結\"\n"
        "        },\n"
        "        \"battle_plan\": {\n"
        "            \"sniper_points\": {\n"
        "                \"ideal_buy\": \"理想買入點：XX元（在MA5附近）\",\n"
        "                \"secondary_buy\": \"次優買入點：XX元（在MA10附近）\",\n"
        "                \"stop_loss\": \"止損位：XX元（跌破MA20或X%）\",\n"
        "                \"take_profit\": \"目標位：XX元（前高/整數關口）\"\n"
        "            },\n"
        "            \"position_strategy\": {\n"
        "                \"suggested_position\": \"建議倉位：X成\",\n"
        "                \"entry_plan\": \"分批建倉策略描述\",\n"
        "                \"risk_control\": \"風控策略描述\"\n"
        "            },\n"
        "            \"action_checklist\": [\n"
        "                \"✅/⚠️/❌ 檢查項1：多頭排列\",\n"
        "                \"✅/⚠️/❌ 檢查項2：乖離率合理（強勢趨勢可放寬）\",\n"
        "                \"✅/⚠️/❌ 檢查項3：量能配合\",\n"
        "                \"✅/⚠️/❌ 檢查項4：無重大利空\",\n"
        "                \"✅/⚠️/❌ 檢查項5：籌碼健康\",\n"
        "                \"✅/⚠️/❌ 檢查項6：PE估值合理\"\n"
        "            ]\n"
        "        }\n"
        "    },\n"
        "    \"analysis_summary\": \"100字綜合分析摘要\",\n"
        "    \"key_points\": \"3-5個核心看點，逗號分隔\",\n"
        "    \"risk_warning\": \"風險提示\",\n"
        "    \"buy_reason\": \"操作理由，引用交易理念\",\n"
        "    \"trend_analysis\": \"走勢形態分析\",\n"
        "    \"short_term_outlook\": \"短期1-3日展望\",\n"
        "    \"medium_term_outlook\": \"中期1-2週展望\",\n"
        "    \"technical_analysis\": \"技術面綜合分析\",\n"
        "    \"ma_analysis\": \"均線系統分析\",\n"
        "    \"volume_analysis\": \"量能分析\",\n"
        "    \"pattern_analysis\": \"K線形態分析\",\n"
        "    \"fundamental_analysis\": \"基本面分析\",\n"
        "    \"sector_position\": \"板塊行業分析\",\n"
        "    \"company_highlights\": \"公司亮點/風險\",\n"
        "    \"news_summary\": \"新聞摘要\",\n"
        "    \"market_sentiment\": \"市場情緒\",\n"
        "    \"hot_topics\": \"相關熱點\",\n"
        "    \"search_performed\": true,\n"
        "    \"data_sources\": \"數據來源說明\"\n"
        "}\n"
        "```\n\n"
        "## 評分標準\n\n"
        "### 強烈買入（80-100分）：\n"
        "- ✅ 多頭排列：MA5 > MA10 > MA20\n"
        "- ✅ 低乖離率：<2%，最佳買點\n"
        "- ✅ 縮量回調或放量突破\n"
        "- ✅ 籌碼集中健康\n"
        "- ✅ 消息面有利好催化\n\n"
        "### 買入（60-79分）：\n"
        "- ✅ 多頭排列或弱勢多頭\n"
        "- ✅ 乖離率 <5%\n"
        "- ✅ 量能正常\n"
        "- ⚪ 允許一項次要條件不滿足\n\n"
        "### 觀望（40-59分）：\n"
        "- ⚠️ 乖離率 >5%（追高風險）\n"
        "- ⚠️ 均線纏繞趨勢不明\n"
        "- ⚠️ 有風險事件\n\n"
        "### 賣出/減倉（0-39分）：\n"
        "- ❌ 空頭排列\n"
        "- ❌ 跌破MA20\n"
        "- ❌ 放量下跌\n"
        "- ❌ 重大利空\n\n"
        "## 決策儀表板核心原則\n\n"
        "1. **核心結論先行**：一句話說清該買該賣\n"
        "2. **分持倉建議**：空倉者和持倉者給不同建議\n"
        "3. **精確狙擊點**：必須給出具體價格，不說模糊的話\n"
        "4. **檢查清單可視化**：用 ✅⚠️❌ 明確顯示每項檢查結果\n"
        "5. **風險優先級**：輿情中的風險點要醒目標出"
    )

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Analyzer via LiteLLM."""
        self._router = None
        self._litellm_available = False
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys), AI analysis will be unavailable")

    def _has_channel_config(self, config: Config) -> bool:
        """Check if multi-channel config is active."""
        return bool(config.llm_model_list) and not all(
            e.get('model_name', '').startswith('__legacy_') for e in config.llm_model_list
        )

    def _init_litellm(self) -> None:
        """Initialize litellm Router from channels / YAML / legacy keys."""
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model:
            logger.warning("Analyzer LLM: LITELLM_MODEL not configured")
            return

        self._litellm_available = True

        if self._has_channel_config(config):
            model_list = config.llm_model_list
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            unique_models = list(dict.fromkeys(
                e['litellm_params']['model'] for e in model_list
            ))
            logger.info(
                f"Analyzer LLM: Router initialized from channels/YAML — "
                f"{len(model_list)} deployment(s), models: {unique_models}"
            )
            return

        keys = get_api_keys_for_model(litellm_model, config)

        if len(keys) > 1:
            extra_params = extra_litellm_params(litellm_model, config)
            legacy_model_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {
                        "model": litellm_model,
                        "api_key": k,
                        **extra_params,
                    },
                }
                for k in keys
            ]
            self._router = Router(
                model_list=legacy_model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            logger.info(
                f"Analyzer LLM: Legacy Router initialized with {len(keys)} keys "
                f"for {litellm_model}"
            )
        elif keys:
            logger.info(f"Analyzer LLM: litellm initialized (model={litellm_model})")
        else:
            logger.info(
                f"Analyzer LLM: litellm initialized (model={litellm_model}, "
                f"API key from environment)"
            )

    def is_available(self) -> bool:
        """Check if LiteLLM is properly configured with at least one API key."""
        return self._router is not None or self._litellm_available

    def _call_litellm(self, prompt: str, generation_config: dict) -> Tuple[str, str]:
        """Call LLM via litellm with fallback across configured models."""
        config = get_config()
        max_tokens = (
            generation_config.get('max_output_tokens')
            or generation_config.get('max_tokens')
            or 8192
        )
        temperature = generation_config.get('temperature', 0.7)

        models_to_try = [config.litellm_model] + (config.litellm_fallback_models or [])
        models_to_try = [m for m in models_to_try if m]

        use_channel_router = self._has_channel_config(config)

        last_error = None
        for model in models_to_try:
            try:
                model_short = model.split("/")[-1] if "/" in model else model
                call_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                extra = get_thinking_extra_body(model_short)
                if extra:
                    call_kwargs["extra_body"] = extra

                if use_channel_router and self._router:
                    response = self._router.completion(**call_kwargs)
                elif self._router and model == config.litellm_model:
                    response = self._router.completion(**call_kwargs)
                else:
                    keys = get_api_keys_for_model(model, config)
                    if keys:
                        call_kwargs["api_key"] = keys[0]
                    call_kwargs.update(extra_litellm_params(model, config))
                    response = litellm.completion(**call_kwargs)

                if response and response.choices and response.choices[0].message.content:
                    return (response.choices[0].message.content, model)
                raise ValueError("LLM returned empty response")

            except Exception as e:
                logger.warning(f"[LiteLLM] {model} failed: {e}")
                last_error = e
                continue

        raise Exception(f"All LLM models failed (tried {len(models_to_try)} model(s)). Last error: {last_error}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Public entry point for free-form text generation."""
        try:
            result = self._call_litellm(
                prompt,
                generation_config={"max_tokens": max_tokens, "temperature": temperature},
            )
            return result[0] if isinstance(result, tuple) else result
        except Exception as exc:
            logger.error("[generate_text] LLM call failed: %s", exc)
            return None

    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """分析單隻股票"""
        code = context.get('code', 'Unknown')
        config = get_config()
        
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 請求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震盪',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未啟用（未配置 API Key）',
                risk_warning='請配置 LLM API Key（GEMINI_API_KEY/ANTHROPIC_API_KEY/OPENAI_API_KEY）後重試',
                success=False,
                error_message='LLM API Key 未配置',
                model_used=None,
            )
        
        try:
            prompt = self._format_prompt(context, name, news_context)
            
            config = get_config()
            model_name = config.litellm_model or "unknown"
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")
            logger.info(f"[LLM配置] Prompt 長度: {len(prompt)} 字元")
            logger.info(f"[LLM配置] 是否包含新聞: {'是' if news_context else '否'}")

            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt 預覽]\n{prompt_preview}")
            logger.debug(f"=== 完整 Prompt ({len(prompt)}字元) ===\n{prompt}\n=== End Prompt ===")

            generation_config = {
                "temperature": config.gemini_temperature,
                "max_output_tokens": 8192,
            }

            logger.info(f"[LLM調用] 開始調用 {model_name}...")

            start_time = time.time()
            response_text, model_used = self._call_litellm(prompt, generation_config)
            elapsed = time.time() - start_time

            logger.info(f"[LLM返回] {model_name} 響應成功, 耗時 {elapsed:.2f}s, 響應長度 {len(response_text)} 字元")
            
            response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            logger.info(f"[LLM返回 預覽]\n{response_preview}")
            logger.debug(f"=== {model_name} 完整響應 ({len(response_text)}字元) ===\n{response_text}\n=== End Response ===")
            
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            result.market_snapshot = self._build_market_snapshot(context)
            result.model_used = model_used

            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 評分 {result.sentiment_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失敗: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震盪',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析過程出錯: {str(e)[:100]}',
                risk_warning='分析失敗，請稍後重試或手動分析',
                success=False,
                error_message=str(e),
                model_used=None,
            )
    
    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """格式化分析提示詞（決策儀表板 v2.0）"""
        code = context.get('code', 'Unknown')
        
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            
        today = context.get('today', {})
        
        prompt_parts = []
        prompt_parts.append("# 決策儀表板分析請求\n")
        prompt_parts.append("## 📊 股票基礎信息\n")
        prompt_parts.append("| 項目 | 數據 |\n")
        prompt_parts.append("|------|------|\n")
        prompt_parts.append(f"| 股票代碼 | **{code}** |\n")
        prompt_parts.append(f"| 股票名稱 | **{stock_name}** |\n")
        prompt_parts.append(f"| 分析日期 | {context.get('date', '未知')} |\n\n")
        prompt_parts.append("---\n\n")
        
        prompt_parts.append("## 📈 技術面數據\n\n")
        prompt_parts.append("### 今日行情\n")
        prompt_parts.append("| 指標 | 數值 |\n")
        prompt_parts.append("|------|------|\n")
        prompt_parts.append(f"| 收盤價 | {today.get('close', 'N/A')} 元 |\n")
        prompt_parts.append(f"| 開盤價 | {today.get('open', 'N/A')} 元 |\n")
        prompt_parts.append(f"| 最高價 | {today.get('high', 'N/A')} 元 |\n")
        prompt_parts.append(f"| 最低價 | {today.get('low', 'N/A')} 元 |\n")
        prompt_parts.append(f"| 漲跌幅 | {today.get('pct_chg', 'N/A')}% |\n")
        prompt_parts.append(f"| 成交量 | {self._format_volume(today.get('volume'))} |\n")
        prompt_parts.append(f"| 成交額 | {self._format_amount(today.get('amount'))} |\n\n")
        
        prompt_parts.append("### 均線系統（關鍵判斷指標）\n")
        prompt_parts.append("| 均線 | 數值 | 說明 |\n")
        prompt_parts.append("|------|------|------|\n")
        prompt_parts.append(f"| MA5 | {today.get('ma5', 'N/A')} | 短期趨勢線 |\n")
        prompt_parts.append(f"| MA10 | {today.get('ma10', 'N/A')} | 中短期趨勢線 |\n")
        prompt_parts.append(f"| MA20 | {today.get('ma20', 'N/A')} | 中期趨勢線 |\n")
        prompt_parts.append(f"| 均線形態 | {context.get('ma_status', '未知')} | 多頭/空頭/纏繞 |\n\n")
        
        if 'realtime' in context:
            rt = context['realtime']
            prompt_parts.append("### 實時行情增強數據\n")
            prompt_parts.append("| 指標 | 數值 | 解讀 |\n")
            prompt_parts.append("|------|------|------|\n")
            prompt_parts.append(f"| 當前價格 | {rt.get('price', 'N/A')} 元 | |\n")
            prompt_parts.append(f"| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |\n")
            prompt_parts.append(f"| **換手率** | **{rt.get('turnover_rate', 'N/A')}%** | |\n")
            prompt_parts.append(f"| 本益比(動態) | {rt.get('pe_ratio', 'N/A')} | |\n")
            prompt_parts.append(f"| 股價淨值比 | {rt.get('pb_ratio', 'N/A')} | |\n")
            prompt_parts.append(f"| 總市值 | {self._format_amount(rt.get('total_mv'))} | |\n")
            prompt_parts.append(f"| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |\n")
            prompt_parts.append(f"| 60日漲跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表現 |\n\n")
        
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt_parts.append("### 籌碼分佈數據（效率指標）\n")
            prompt_parts.append("| 指標 | 數值 | 健康標準 |\n")
            prompt_parts.append("|------|------|----------|\n")
            prompt_parts.append(f"| **獲利比例** | **{profit_ratio:.1%}** | 70-90%時警惕 |\n")
            prompt_parts.append(f"| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | 現價應高於5-15% |\n")
            prompt_parts.append(f"| 90%籌碼集中度 | {chip.get('concentration_90', 0):.2%} | <15%為集中 |\n")
            prompt_parts.append(f"| 70%籌碼集中度 | {chip.get('concentration_70', 0):.2%} | |\n")
            prompt_parts.append(f"| 籌碼狀態 | {chip.get('chip_status', '未知')} | |\n\n")
        
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 超過5%，嚴禁追高！" if trend.get('bias_ma5', 0) > 5 else "✅ 安全範圍"
            prompt_parts.append("### 趨勢分析預判（基於交易理念）\n")
            prompt_parts.append("| 指標 | 數值 | 判定 |\n")
            prompt_parts.append("|------|------|------|\n")
            prompt_parts.append(f"| 趨勢狀態 | {trend.get('trend_status', '未知')} | |\n")
            prompt_parts.append(f"| 均線排列 | {trend.get('ma_alignment', '未知')} | MA5>MA10>MA20為多頭 |\n")
            prompt_parts.append(f"| 趨勢強度 | {trend.get('trend_strength', 0)}/100 | |\n")
            prompt_parts.append(f"| **乖離率(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |\n")
            prompt_parts.append(f"| 乖離率(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |\n")
            prompt_parts.append(f"| 量能狀態 | {trend.get('volume_status', '未知')} | {trend.get('volume_trend', '')} |\n")
            prompt_parts.append(f"| 系統信號 | {trend.get('buy_signal', '未知')} | |\n")
            prompt_parts.append(f"| 系統評分 | {trend.get('signal_score', 0)}/100 | |\n\n")
            
            prompt_parts.append("#### 系統分析理由\n")
            prompt_parts.append("**買入理由**：\n")
            if trend.get('signal_reasons'):
                for r in trend.get('signal_reasons'):
                    prompt_parts.append(f"- {r}\n")
            else:
                prompt_parts.append("- 無\n")
                
            prompt_parts.append("\n**風險因素**：\n")
            if trend.get('risk_factors'):
                for r in trend.get('risk_factors'):
                    prompt_parts.append(f"- {r}\n")
            else:
                prompt_parts.append("- 無\n")
            prompt_parts.append("\n")
        
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt_parts.append("### 量價變化\n")
            prompt_parts.append(f"- 成交量較昨日變化：{volume_change}倍\n")
            prompt_parts.append(f"- 價格較昨日變化：{context.get('price_change_ratio', 'N/A')}%\n\n")
        
        prompt_parts.append("---\n\n## 📰 輿情情報\n\n")
        
        if news_context:
            prompt_parts.append(f"以下是 **{stock_name}({code})** 近7日的新聞搜索結果，請重點提取：\n")
            prompt_parts.append("1. 🚨 **風險警報**：減持、處罰、利空\n")
            prompt_parts.append("2. 🎯 **利好催化**：業績、合同、政策\n")
            prompt_parts.append("3. 📊 **業績預期**：年報預告、業績快報\n\n")
            prompt_parts.append("```\n")
            prompt_parts.append(f"{news_context}\n")
            prompt_parts.append("```\n\n")
        else:
            prompt_parts.append("未搜索到該股票近期的相關新聞。請主要依據技術面數據進行分析。\n\n")

        if context.get('data_missing'):
            prompt_parts.append("⚠️ **數據缺失警告**\n")
            prompt_parts.append("由於接口限制，當前無法獲取完整的實時行情和技術指標數據。\n")
            prompt_parts.append("請 **忽略上述表格中的 N/A 數據**，重點依據 **【📰 輿情情報】** 中的新聞進行基本面和情緒面分析。\n")
            prompt_parts.append("在回答技術面問題（如均線、乖離率）時，請直接說明「數據缺失，無法判斷」，**嚴禁編造數據**。\n\n")

        prompt_parts.append("---\n\n## ✅ 分析任務\n\n")
        prompt_parts.append(f"請為 **{stock_name}({code})** 生成【決策儀表板】，嚴格按照 JSON 格式輸出。\n")
        
        if context.get('is_index_etf'):
            prompt_parts.append("> ⚠️ **指數/ETF 分析約束**：該標的為指數跟蹤型 ETF 或市場指數。\n")
            prompt_parts.append("> - 風險分析僅關注：**指數走勢、跟蹤誤差、市場流動性**\n")
            prompt_parts.append("> - 嚴禁將基金公司的訴訟、聲譽、高管變動納入風險警報\n")
            prompt_parts.append("> - 業績預期基於**指數成分股整體表現**，而非基金公司財報\n")
            prompt_parts.append("> - `risk_alerts` 中不得出現基金管理人相關的公司經營風險\n\n")

        prompt_parts.append("### ⚠️ 重要：輸出正確的股票名稱格式\n")
        prompt_parts.append("正確的股票名稱格式為「股票名稱（股票代碼）」，例如「貴州茅台（600519）」。\n")
        prompt_parts.append(f"如果上方顯示的股票名稱為\"股票{code}\"或不正確，請在分析開頭**明確輸出該股票的正確中文全稱**。\n\n")

        prompt_parts.append("### 重點關注（必須明確回答）：\n")
        prompt_parts.append("1. ❓ 是否滿足 MA5>MA10>MA20 多頭排列？\n")
        prompt_parts.append("2. ❓ 當前乖離率是否在安全範圍內（<5%）？—— 超過5%必須標註\"嚴禁追高\"\n")
        prompt_parts.append("3. ❓ 量能是否配合（縮量回調/放量突破）？\n")
        prompt_parts.append("4. ❓ 籌碼結構是否健康？\n")
        prompt_parts.append("5. ❓ 消息面有無重大利空？（減持、處罰、業績變臉等）\n\n")

        prompt_parts.append("### 決策儀表板要求：\n")
        prompt_parts.append("- **股票名稱**：必須輸出正確的中文全稱（如\"貴州茅台\"而非\"股票600519\"）\n")
        prompt_parts.append("- **核心結論**：一句話說清該買/該賣/該等\n")
        prompt_parts.append("- **持倉分類建議**：空倉者怎麼做 vs 持倉者怎麼做\n")
        prompt_parts.append("- **具體狙擊點位**：買入價、止損價、目標價（精確到分）\n")
        prompt_parts.append("- **檢查清單**：每項用 ✅/⚠️/❌ 標記\n\n")
        prompt_parts.append("請輸出完整的 JSON 格式決策儀表板。")
        
        return "".join(prompt_parts)
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """格式化成交量顯示"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 億股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 萬股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """格式化成交額顯示"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 億元"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 萬元"
        else:
            return f"{amount:.0f} 元"

    def _format_percent(self, value: Optional[float]) -> str:
        """格式化百分比顯示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_price(self, value: Optional[float]) -> str:
        """格式化價格顯示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """構建當日行情快照（展示用）"""
        today = context.get('today', {}) or {}
        realtime = context.get('realtime', {}) or {}
        yesterday = context.get('yesterday', {}) or {}

        prev_close = yesterday.get('close')
        close = today.get('close')
        high = today.get('high')
        low = today.get('low')

        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                change_amount = None

        snapshot = {
            "date": context.get('date', '未知'),
            "close": self._format_price(close),
            "open": self._format_price(today.get('open')),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get('pct_chg')),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get('volume')),
            "amount": self._format_amount(today.get('amount')),
        }

        if realtime:
            snapshot.update({
                "price": self._format_price(realtime.get('price')),
                "volume_ratio": realtime.get('volume_ratio', 'N/A'),
                "turnover_rate": self._format_percent(realtime.get('turnover_rate')),
                "source": getattr(realtime.get('source'), 'value', realtime.get('source', 'N/A')),
            })

        return snapshot

    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """
        解析 Gemini 響應（決策儀表板版）
        
        嘗試從響應中提取 JSON 格式的分析結果，包含 dashboard 字段
        如果解析失敗，嘗試智能提取或返回默認結果
        """
        try:
            # 清理響應文本：移除 markdown 代碼塊標記
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # 嘗試找到 JSON 內容
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # 嘗試修復常見的 JSON 問題
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)
                
                # 提取 dashboard 數據
                dashboard = data.get('dashboard', None)

                # 優先使用 AI 返回的股票名稱（如果原名稱無效或包含代碼）
                ai_stock_name = data.get('stock_name')
                if ai_stock_name and (name.startswith('股票') or name == code or 'Unknown' in name):
                    name = ai_stock_name

                # 解析所有字段，使用默認值防止缺失
                # 解析 decision_type，如果沒有則根據 operation_advice 推斷
                decision_type = data.get('decision_type', '')
                if not decision_type:
                    op = data.get('operation_advice', '持有')
                    if op in ['買入', '加倉', '強烈買入']:
                        decision_type = 'buy'
                    elif op in ['賣出', '減倉', '強烈賣出']:
                        decision_type = 'sell'
                    else:
                        decision_type = 'hold'
                
                return AnalysisResult(
                    code=code,
                    name=name,
                    # 核心指標
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', '震盪'),
                    operation_advice=data.get('operation_advice', '持有'),
                    decision_type=decision_type,
                    confidence_level=data.get('confidence_level', '中'),
                    # 決策儀表板
                    dashboard=dashboard,
                    # 走勢分析
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # 技術面
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # 基本面
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # 情緒面/消息面
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # 綜合
                    analysis_summary=data.get('analysis_summary', '分析完成'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # 元數據
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', '技術面數據'),
                    success=True,
                )
            else:
                # 沒有找到 JSON，嘗試從純文本中提取信息
                logger.warning(f"無法從響應中提取 JSON，使用原始文本分析")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失敗: {e}，嘗試從文本提取")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修復常見的 JSON 格式問題"""
        import re
        
        # 移除註釋
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修復尾隨逗號
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 確保布爾值是小寫
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        # fix by json-repair
        json_str = repair_json(json_str)
        
        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """從純文本響應中盡可能提取分析信息"""
        # 嘗試識別關鍵詞來判斷情緒
        sentiment_score = 50
        trend = '震盪'
        advice = '持有'
        
        text_lower = response_text.lower()
        
        # 簡單的情緒識別
        positive_keywords = ['看多', '買入', '上漲', '突破', '強勢', '利好', '加倉', 'bullish', 'buy']
        negative_keywords = ['看空', '賣出', '下跌', '跌破', '弱勢', '利空', '減倉', 'bearish', 'sell']
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = '看多'
            advice = '買入'
            decision_type = 'buy'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = '看空'
            advice = '賣出'
            decision_type = 'sell'
        else:
            decision_type = 'hold'
        
        # 截取前500字符作為摘要
        summary = response_text[:500] if response_text else '無分析結果'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            decision_type=decision_type,
            confidence_level='低',
            analysis_summary=summary,
            key_points='JSON解析失敗，僅供參考',
            risk_warning='分析結果可能不準確，建議結合其他信息判斷',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多隻股票
        
        注意：為避免 API 速率限制，每次分析之間會有延遲
        
        Args:
            contexts: 上下文數據列表
            delay_between: 每次分析之間的延遲（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒後繼續...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函數
def get_analyzer() -> GeminiAnalyzer:
    """獲取 LLM 分析器實例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 測試代碼
    logging.basicConfig(level=logging.DEBUG)
    
    # 模擬上下文數據
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': '多頭排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析測試 ===")
        result = analyzer.analyze(test_context)
        print(f"分析結果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳過測試")
