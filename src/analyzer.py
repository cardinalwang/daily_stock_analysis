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
    """
    多來源獲取股票中文名稱

    獲取策略（按優先級）：
    1. 從傳入的 context 中獲取（realtime 數據）
    2. 從靜態映射表 STOCK_NAME_MAP 獲取
    3. 從 DataFetcherManager 獲取（各數據源）
    4. 返回默認名稱（股票+代碼）

    Args:
        stock_code: 股票代碼
        context: 分析上下文（可選）
        data_manager: DataFetcherManager 實例（可選）

    Returns:
        股票中文名稱
    """
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
    """
    AI 分析結果數據類 - 決策儀表板版

    封裝 Gemini 返回的分析結果，包含決策儀表板和詳細分析
    """
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
            'dashboard': self.dashboard,  # 決策儀表板數據
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
            '買入': '🟢',
            '加倉': '🟢',
            '強烈買入': '💚',
            '持有': '🟡',
            '觀望': '⚪',
            '減倉': '🟠',
            '賣出': '🔴',
            '強烈賣出': '❌',
        }
        advice = self.operation_advice or ''
        # Direct match first
        if advice in emoji_map:
            return emoji_map[advice]
        # Handle compound advice like "賣出/觀望" — use the first part
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        # Score-based fallback
        score = self.sentiment_score
        if score >= 80:
            return '💚'
        elif score >= 65:
            return '🟢'
        elif score >= 55:
            return '🟡'
        elif score >= 45:
            return '⚪'
        elif score >= 35:
            return '🟠'
        else:
            return '🔴'

    def get_confidence_stars(self) -> str:
        """返回置信度星級"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器

    職責：
    1. 調用 Google Gemini API 進行股票分析
    2. 結合預先搜索的新聞和技術面數據生成分析報告
    3. 解析 AI 返回的 JSON 格式結果

    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """

    # ========================================
    # 系統提示詞 - 決策儀表板 v2.0
    # ========================================
    # 輸出格式升級：從簡單信號升級為決策儀表板
    # 核心模塊：核心結論 + 數據透視 + 輿情情報 + 作戰計畫
    # ========================================

    SYSTEM_PROMPT = """你是一位專注於趨勢交易的投資分析師，負責生成專業的【決策儀表板】分析報告。

## 核心交易理念（必須嚴格遵守）

### 1. 嚴進策略（不追高）
- **絕對不追高**：當股價偏離 MA5 超過 5% 時，堅決不買入
- **乖離率公式**：(現價 - MA5) / MA5 × 100%
- 乖離率 < 2%：最佳買點區間
- 乖離率 2-5%：可小倉介入
- 乖離率 > 5%：嚴禁追高！直接判定為"觀望"

### 2. 趨勢交易（順勢而為）
- **多頭排列必須條件**：MA5 > MA10 > MA20
- 只做多頭排列的股票，空頭排列堅決不碰
- 均線發散上行優於均線粘合
- 趨勢強度判斷：看均線間距是否在擴大

### 3. 效率優先（籌碼結構）
- 關注籌碼集中度：90%集中度 < 15% 表示籌碼集中
- 獲利比例分析：70-90% 獲利盤時需警惕獲利回吐
- 平均成本與現價關係：現價高於平均成本 5-15% 為健康

### 4. 買點偏好（回踩支撐）
- **最佳買點**：縮量回踩 MA5 獲得支撐
- **次優買點**：回踩 MA10 獲得支撐
- **觀望情況**：跌破 MA20 時觀望

### 5. 風險排查重點
- 減持公告（股東、高管減持）
- 業績預虧/大幅下滑
- 監管處罰/立案調查
- 行業政策利空
- 大額解禁

### 6. 估值關注（PE/PB）
- 分析時請關注本益比（PE）是否合理
- PE 明顯偏高時（如遠超行業平均或歷史均值），需在風險點中說明
- 高成長股可適當容忍較高 PE，但需有業績支撐

### 7. 強勢趨勢股放寬
- 強勢趨勢股（多頭排列且趨勢強度高、量能配合）可適當放寬乖離率要求
- 此類股票可輕倉追蹤，但仍需設置止損，不盲目追高

## 輸出格式：決策儀表板 JSON

請嚴格按照以下 JSON 格式輸出，這是一個完整的【決策儀表板】：

```json
{
    "stock_name": "股票中文名稱",
    "sentiment_score": 0-100整数,
    "trend_prediction": "強烈看多/看多/震盪/看空/強烈看空",
    "operation_advice": "買入/加倉/持有/減倉/賣出/觀望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句話核心結論（30字以內，直接告訴用戶做什麼）",
            "signal_type": "🟢買入信號/🟡持有觀望/🔴賣出信號/⚠️風險警告",
            "time_sensitivity": "立即行動/今日內/本週內/不急",
            "position_advice": {
                "no_position": "空倉者建議：具體操作指引",
                "has_position": "持倉者建議：具體操作指引"
            }
        },

        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均線排列狀態描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 当前价格数值,
                "ma5": MA5数值,
                "ma10": MA10数值,
                "ma20": MA20数值,
                "bias_ma5": 乖離率百分比數值,
                "bias_status": "安全/警戒/危險",
                "support_level": 支撐位價格,
                "resistance_level": 壓力位價格
            },
            "volume_analysis": {
                "volume_ratio": 量比數值,
                "volume_status": "放量/縮量/平量",
                "turnover_rate": 換手率百分比,
                "volume_meaning": "量能含義解讀（如：縮量回調表示拋壓減輕）"
            },
            "chip_structure": {
                "profit_ratio": 獲利比例,
                "avg_cost": 平均成本,
                "concentration": 籌碼集中度,
                "chip_health": "健康/一般/警惕"
            }
        },

        "intelligence": {
            "latest_news": "【最新消息】近期重要新聞摘要",
            "risk_alerts": ["風險點1：具體描述", "風險點2：具體描述"],
            "positive_catalysts": ["利好1：具體描述", "利好2：具體描述"],
            "earnings_outlook": "業績預期分析（基於年報預告、業績快報等）",
            "sentiment_summary": "輿情情緒一句話總結"
        },

        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想買入點：XX元（在MA5附近）",
                "secondary_buy": "次優買入點：XX元（在MA10附近）",
                "stop_loss": "止損位：XX元（跌破MA20或X%）",
                "take_profit": "目標位：XX元（前高/整數關口）"
            },
            "position_strategy": {
                "suggested_position": "建議倉位：X成",
                "entry_plan": "分批建倉策略描述",
                "risk_control": "風控策略描述"
            },
            "action_checklist": [
                "✅/⚠️/❌ 檢查項1：多頭排列",
                "✅/⚠️/❌ 檢查項2：乖離率合理（強勢趨勢可放寬）",
                "✅/⚠️/❌ 檢查項3：量能配合",
                "✅/⚠️/❌ 檢查項4：無重大利空",
                "✅/⚠️/❌ 檢查項5：籌碼健康",
                "✅/⚠️/❌ 檢查項6：PE估值合理"
            ]
        }
    },

    "analysis_summary": "100字綜合分析摘要",
    "key_points": "3-5個核心看點，逗號分隔",
    "risk_warning": "風險提示",
    "buy_reason": "操作理由，引用交易理念",

    "trend_analysis": "走勢形態分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2週展望",
    "technical_analysis": "技術面綜合分析",
    "ma_analysis": "均線系統分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K線形態分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板塊行業分析",
    "company_highlights": "公司亮點/風險",
    "news_summary": "新聞摘要",
    "market_sentiment": "市場情緒",
    "hot_topics": "相關熱點",

    "search_performed": true/false,
    "data_sources": "數據來源說明"
}
