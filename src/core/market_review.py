# -*- coding: utf-8 -*-
"""
===================================
股票智能分析系統 - 大盤復盤模塊（支持 A 股 / 美股）
===================================

職責：
1. 根據 MARKET_REVIEW_REGION 配置選擇市場區域（cn / us / both）
2. 執行大盤復盤分析並生成復盤報告
3. 保存和發送復盤報告
"""

import logging
from datetime import datetime
from typing import Optional

from src.config import get_config
from src.notification import NotificationService
from src.market_analyzer import MarketAnalyzer
from src.search_service import SearchService
from src.analyzer import GeminiAnalyzer


logger = logging.getLogger(__name__)


def run_market_review(
    notifier: NotificationService,
    analyzer: Optional[GeminiAnalyzer] = None,
    search_service: Optional[SearchService] = None,
    send_notification: bool = True,
    merge_notification: bool = False,
    override_region: Optional[str] = None,
) -> Optional[str]:
    """
    執行大盤復盤分析

    Args:
        notifier: 通知服務
        analyzer: AI分析器（可選）
        search_service: 搜索服務（可選）
        send_notification: 是否發送通知
        merge_notification: 是否合併推送（跳過本次推送，由 main 層合併個股+大盤後統一發送，Issue #190）
        override_region: 覆蓋 config 的 market_review_region（Issue #373 交易日過濾後有效子集）

    Returns:
        復盤報告文本
    """
    logger.info("開始執行大盤復盤分析...")
    config = get_config()
    region = (
        override_region
        if override_region is not None
        else (getattr(config, 'market_review_region', 'cn') or 'cn')
    )
    if region not in ('cn', 'us', 'both'):
        region = 'cn'

    try:
        if region == 'both':
            # 順序執行 A 股 + 美股，合併報告
            cn_analyzer = MarketAnalyzer(
                search_service=search_service, analyzer=analyzer, region='cn'
            )
            us_analyzer = MarketAnalyzer(
                search_service=search_service, analyzer=analyzer, region='us'
            )
            logger.info("生成 A 股大盤復盤報告...")
            cn_report = cn_analyzer.run_daily_review()
            logger.info("生成美股大盤復盤報告...")
            us_report = us_analyzer.run_daily_review()
            review_report = ''
            if cn_report:
                review_report = f"# A股大盤復盤\n\n{cn_report}"
            if us_report:
                if review_report:
                    review_report += "\n\n---\n\n> 以下為美股大盤復盤\n\n"
                review_report += f"# 美股大盤復盤\n\n{us_report}"
            if not review_report:
                review_report = None
        else:
            market_analyzer = MarketAnalyzer(
                search_service=search_service,
                analyzer=analyzer,
                region=region,
            )
            review_report = market_analyzer.run_daily_review()
        
        if review_report:
            # 保存報告到文件
            date_str = datetime.now().strftime('%Y%m%d')
            report_filename = f"market_review_{date_str}.md"
            filepath = notifier.save_report_to_file(
                f"# □□□□□n\n{review_report}", 
                report_filename
            )
            logger.info(f"大盤復盤報告已保存: {filepath}")
            
            # 推送通知（合併模式下跳過，由 main 層統一發送）
            if merge_notification and send_notification:
                logger.info("合併推送模式：跳過大盤復盤單獨推送，將在個股+大盤復盤後統一發送")
            elif send_notification and notifier.is_available():
                # 添加標題
                report_content = f"□□□□□n\n{review_report}"

                success = notifier.send(report_content, email_send_to_all=True)
                if success:
                    logger.info("大盤復盤推送成功")
                else:
                    logger.warning("大盤復盤推送失敗")
            elif not send_notification:
                logger.info("已跳過推送通知 (--no-notify)")
            
            return review_report
        
    except Exception as e:
        logger.error(f"大盤復盤分析失敗: {e}")
    
    return None
