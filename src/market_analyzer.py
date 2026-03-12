
"""
===================================
A股自選股智能分析系統 - 主調度程序
===================================

職責：
1. 協調各模塊完成股票分析流程
2. 實現低並發的線程池調度
3. 全局異常處理，確保單股失敗不影響整體
4. 提供命令行入口

使用方式：
    python main.py              # 正常運行
    python main.py --debug      # 調試模式
    python main.py --dry-run    # 僅獲取數據不分析

交易理念（已融入分析）：
- 嚴進策略：不追高，乖離率 > 5% 不買入
- 趨勢交易：只做 MA5>MA10>MA20 多頭排列
- 效率優先：關注籌碼集中度好的股票
- 買點偏好：縮量回踩 MA5/MA10 支撐
"""
import os
from src.config import setup_env
setup_env()

# 代理配置 - 通過 USE_PROXY 環境變量控制，默認關閉
# GitHub Actions 環境自動跳過代理配置
if os.getenv("GITHUB_ACTIONS") != "true" and os.getenv("USE_PROXY", "false").lower() == "true":
    # 本地開發環境，啟用代理（可在 .env 中配置 PROXY_HOST 和 PROXY_PORT）
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "10809")
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from data_provider.base import canonical_stock_code
from src.core.pipeline import StockAnalysisPipeline
from src.core.market_review import run_market_review
from src.webui_frontend import prepare_webui_frontend_assets
from src.config import get_config, Config
from src.logging_config import setup_logging


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='A股自選股智能分析系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py                    # 正常運行
  python main.py --debug            # 調試模式
  python main.py --dry-run          # 僅獲取數據，不進行 AI 分析
  python main.py --stocks 600519,000001  # 指定分析特定股票
  python main.py --no-notify        # 不發送推送通知
  python main.py --single-notify    # 啟用單股推送模式（每分析完一隻立即推送）
  python main.py --schedule         # 啟用定時任務模式
  python main.py --market-review    # 僅運行大盤復盤
        '''
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='啟用調試模式，輸出詳細日誌'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='僅獲取數據，不進行 AI 分析'
    )

    parser.add_argument(
        '--stocks',
        type=str,
        help='指定要分析的股票代碼，逗號分隔（覆蓋配置文件）'
    )

    parser.add_argument(
        '--no-notify',
        action='store_true',
        help='不發送推送通知'
    )

    parser.add_argument(
        '--single-notify',
        action='store_true',
        help='啟用單股推送模式：每分析完一隻股票立即推送，而不是匯總推送'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='並發線程數（默認使用配置值）'
    )

    parser.add_argument(
        '--schedule',
        action='store_true',
        help='啟用定時任務模式，每日定時執行'
    )

    parser.add_argument(
        '--no-run-immediately',
        action='store_true',
        help='定時任務啟動時不立即執行一次'
    )

    parser.add_argument(
        '--market-review',
        action='store_true',
        help='僅運行大盤復盤分析'
    )

    parser.add_argument(
        '--no-market-review',
        action='store_true',
        help='跳過大盤復盤分析'
    )

    parser.add_argument(
        '--force-run',
        action='store_true',
        help='跳過交易日檢查，強制執行全量分析（Issue #373）'
    )

    parser.add_argument(
        '--webui',
        action='store_true',
        help='啟動 Web 管理界面'
    )

    parser.add_argument(
        '--webui-only',
        action='store_true',
        help='僅啟動 Web 服務，不執行自動分析'
    )

    parser.add_argument(
        '--serve',
        action='store_true',
        help='啟動 FastAPI 後端服務（同時執行分析任務）'
    )

    parser.add_argument(
        '--serve-only',
        action='store_true',
        help='僅啟動 FastAPI 後端服務，不自動執行分析'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='FastAPI 服務端口（默認 8000）'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='FastAPI 服務監聽地址（默認 0.0.0.0）'
    )

    parser.add_argument(
        '--no-context-snapshot',
        action='store_true',
        help='不保存分析上下文快照'
    )

    # === Backtest ===
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='運行回測（對歷史分析結果進行評估）'
    )

    parser.add_argument(
        '--backtest-code',
        type=str,
        default=None,
        help='僅回測指定股票代碼'
    )

    parser.add_argument(
        '--backtest-days',
        type=int,
        default=None,
        help='回測評估窗口（交易日數，默認使用配置）'
    )

    parser.add_argument(
        '--backtest-force',
        action='store_true',
        help='強制回測（即使已有回測結果也重新計算）'
    )

    return parser.parse_args()


def _compute_trading_day_filter(
    config: Config,
    args: argparse.Namespace,
    stock_codes: List[str],
) -> Tuple[List[str], Optional[str], bool]:
    """
    Compute filtered stock list and effective market review region (Issue #373).

    Returns:
        (filtered_codes, effective_region, should_skip_all)
        - effective_region None = use config default (check disabled)
        - effective_region '' = all relevant markets closed, skip market review
        - should_skip_all: skip entire run when no stocks and no market review to run
    """
    force_run = getattr(args, 'force_run', False)
    if force_run or not getattr(config, 'trading_day_check_enabled', True):
        return (stock_codes, None, False)

    from src.core.trading_calendar import (
        get_market_for_stock,
        get_open_markets_today,
        compute_effective_region,
    )

    open_markets = get_open_markets_today()
    filtered_codes = []
    for code in stock_codes:
        mkt = get_market_for_stock(code)
        if mkt in open_markets or mkt is None:
            filtered_codes.append(code)

    if config.market_review_enabled and not getattr(args, 'no_market_review', False):
        effective_region = compute_effective_region(
            getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
        )
    else:
        effective_region = None

    should_skip_all = (not filtered_codes) and (effective_region or '') == ''
    return (filtered_codes, effective_region, should_skip_all)


def run_full_analysis(
    config: Config,
    args: argparse.Namespace,
    stock_codes: Optional[List[str]] = None
):
    """
    執行完整的分析流程（個股 + 大盤復盤）

    這是定時任務調用的主函數
    """
    try:
        # Issue #529: Hot-reload STOCK_LIST from .env on each scheduled run
        if stock_codes is None:
            config.refresh_stock_list()

        # Issue #373: Trading day filter (per-stock, per-market)
        effective_codes = stock_codes if stock_codes is not None else config.stock_list
        filtered_codes, effective_region, should_skip = _compute_trading_day_filter(
            config, args, effective_codes
        )
        if should_skip:
            logger.info(
                "今日所有相關市場均為非交易日，跳過執行。可使用 --force-run 強制執行。"
            )
            return
        if set(filtered_codes) != set(effective_codes):
            skipped = set(effective_codes) - set(filtered_codes)
            logger.info("今日休市股票已跳過: %s", skipped)
        stock_codes = filtered_codes

        # 命令行參數 --single-notify 覆蓋配置（#55）
        if getattr(args, 'single_notify', False):
            config.single_stock_notify = True

        # Issue #190: 個股與大盤復盤合併推送
        merge_notification = (
            getattr(config, 'merge_email_notification', False)
            and config.market_review_enabled
            and not getattr(args, 'no_market_review', False)
            and not config.single_stock_notify
        )

        # 創建調度器
        save_context_snapshot = None
        if getattr(args, 'no_context_snapshot', False):
            save_context_snapshot = False
        query_id = uuid.uuid4().hex
        pipeline = StockAnalysisPipeline(
            config=config,
            max_workers=args.workers,
            query_id=query_id,
            query_source="cli",
            save_context_snapshot=save_context_snapshot
        )

        # 1. 運行個股分析
        results = pipeline.run(
            stock_codes=stock_codes,
            dry_run=args.dry_run,
            send_notification=not args.no_notify,
            merge_notification=merge_notification
        )

        # Issue #128: 分析間隔 - 在個股分析和大盤分析之間添加延遲
        analysis_delay = getattr(config, 'analysis_delay', 0)
        if (
            analysis_delay > 0
            and config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            logger.info(f"等待 {analysis_delay} 秒後執行大盤復盤（避免API限流）...")
            time.sleep(analysis_delay)

        # 2. 運行大盤復盤（如果啟用且不是僅個股模式）
        market_report = ""
        if (
            config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            review_result = run_market_review(
                notifier=pipeline.notifier,
                analyzer=pipeline.analyzer,
                search_service=pipeline.search_service,
                send_notification=not args.no_notify,
                merge_notification=merge_notification,
                override_region=effective_region,
            )
            # 如果有結果，賦值給 market_report 用於後續飛書文檔生成
            if review_result:
                market_report = review_result

        # Issue #190: 合併推送（個股+大盤復盤）
        if merge_notification and (results or market_report) and not args.no_notify:
            parts = []
            if market_report:
                parts.append(f"# □蠥□□□□n\n{market_report}")
            if results:
                dashboard_content = pipeline.notifier.generate_dashboard_report(results)
                parts.append(f"# □□誨□泧□□□□n\n{dashboard_content}")
            if parts:
                combined_content = "\n\n---\n\n".join(parts)
                if pipeline.notifier.is_available():
                    if pipeline.notifier.send(combined_content, email_send_to_all=True):
                        logger.info("已合併推送（個股+大盤復盤）")
                    else:
                        logger.warning("合併推送失敗")

        # 輸出摘要
        if results:
            logger.info("\n===== 分析結果摘要 =====")
            for r in sorted(results, key=lambda x: x.sentiment_score, reverse=True):
                emoji = r.get_emoji()
                logger.info(
                    f"{emoji} {r.name}({r.code}): {r.operation_advice} | "
                    f"評分 {r.sentiment_score} | {r.trend_prediction}"
                )

        logger.info("\n任務執行完成")

        # === 新增：生成飛書雲文檔 ===
        try:
            from src.feishu_doc import FeishuDocManager

            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and (results or market_report):
                logger.info("正在創建飛書雲文檔...")

                # 1. 準備標題 "01-01 13:01大盤復盤"
                tz_cn = timezone(timedelta(hours=8))
                now = datetime.now(tz_cn)
                doc_title = f"{now.strftime('%Y-%m-%d %H:%M')} 大盤復盤"

                # 2. 準備內容 (拼接個股分析和大盤復盤)
                full_content = ""

                # 添加大盤復盤內容（如果有）
                if market_report:
                    full_content += f"# □蠥□□□□n\n{market_report}\n\n---\n\n"

                # 添加個股決策儀表盤（使用 NotificationService 生成）
                if results:
                    dashboard_content = pipeline.notifier.generate_dashboard_report(results)
                    full_content += f"# □□誨□泧□□□□n\n{dashboard_content}"

                # 3. 創建文檔
                doc_url = feishu_doc.create_daily_doc(doc_title, full_content)
                if doc_url:
                    logger.info(f"飛書雲文檔創建成功: {doc_url}")
                    # 可選：將文檔鏈接也推送到群裡
                    if not args.no_notify:
                        pipeline.notifier.send(f"[{now.strftime('%Y-%m-%d %H:%M')}] 復盤文檔創建成功: {doc_url}")

        except Exception as e:
            logger.error(f"飛書文檔生成失敗: {e}")

        # === Auto backtest ===
        try:
            if getattr(config, 'backtest_enabled', False):
                from src.services.backtest_service import BacktestService

                logger.info("開始自動回測...")
                service = BacktestService()
                stats = service.run_backtest(
                    force=False,
                    eval_window_days=getattr(config, 'backtest_eval_window_days', 10),
                    min_age_days=getattr(config, 'backtest_min_age_days', 14),
                    limit=200,
                )
                logger.info(
                    f"自動回測完成: processed={stats.get('processed')} saved={stats.get('saved')} "
                    f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
                )
        except Exception as e:
            logger.warning(f"自動回測失敗（已忽略）: {e}")

    except Exception as e:
        logger.exception(f"分析流程執行失敗: {e}")


def start_api_server(host: str, port: int, config: Config) -> None:
    """
    在後台線程啟動 FastAPI 服務
    
    Args:
        host: 監聽地址
        port: 監聽端口
        config: 配置對像
    """
    import threading
    import uvicorn

    def run_server():
        level_name = (config.log_level or "INFO").lower()
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            log_level=level_name,
            log_config=None,
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info(f"FastAPI 服務已啟動: http://{host}:{port}")


def _is_truthy_env(var_name: str, default: str = "true") -> bool:
    """Parse common truthy / falsy environment values."""
    value = os.getenv(var_name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}

def start_bot_stream_clients(config: Config) -> None:
    """Start bot stream clients when enabled in config."""
    # 啟動釘釘 Stream 客戶端
    if config.dingtalk_stream_enabled:
        try:
            from bot.platforms import start_dingtalk_stream_background, DINGTALK_STREAM_AVAILABLE
            if DINGTALK_STREAM_AVAILABLE:
                if start_dingtalk_stream_background():
                    logger.info("[Main] Dingtalk Stream client started in background.")
                else:
                    logger.warning("[Main] Dingtalk Stream client failed to start.")
            else:
                logger.warning("[Main] Dingtalk Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install dingtalk-stream")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Dingtalk Stream client: {exc}")

    # 啟動飛書 Stream 客戶端
    if getattr(config, 'feishu_stream_enabled', False):
        try:
            from bot.platforms import start_feishu_stream_background, FEISHU_SDK_AVAILABLE
            if FEISHU_SDK_AVAILABLE:
                if start_feishu_stream_background():
                    logger.info("[Main] Feishu Stream client started in background.")
                else:
                    logger.warning("[Main] Feishu Stream client failed to start.")
            else:
                logger.warning("[Main] Feishu Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install lark-oapi")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Feishu Stream client: {exc}")


def main() -> int:
    """
    主入口函數

    Returns:
        退出碼（0 表示成功）
    """
    # 解析命令行參數
    args = parse_arguments()

    # 加載配置（在設置日誌前加載，以獲取日誌目錄）
    config = get_config()

    # 配置日誌（輸出到控制台和文件）
    setup_logging(log_prefix="stock_analysis", debug=args.debug, log_dir=config.log_dir)

    logger.info("=" * 60)
    logger.info("A股自選股智能分析系統 啟動")
    logger.info(f"運行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 驗證配置
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)

    # 解析股票列表（統一為大寫 Issue #355）
    stock_codes = None
    if args.stocks:
        stock_codes = [canonical_stock_code(c) for c in args.stocks.split(',') if (c or "").strip()]
        logger.info(f"使用命令行指定的股票列表: {stock_codes}")

    # === 處理 --webui / --webui-only 參數，映射到 --serve / --serve-only ===
    if args.webui:
        args.serve = True
    if args.webui_only:
        args.serve_only = True

    # 兼容舊版 WEBUI_ENABLED 環境變量
    if config.webui_enabled and not (args.serve or args.serve_only):
        args.serve = True

    # === 啟動 Web 服務 (如果啟用) ===
    start_serve = (args.serve or args.serve_only) and os.getenv("GITHUB_ACTIONS") != "true"

    # 兼容舊版 WEBUI_HOST/WEBUI_PORT：如果用戶未通過 --host/--port 指定，則使用舊變量
    if start_serve:
        if args.host == '0.0.0.0' and os.getenv('WEBUI_HOST'):
            args.host = os.getenv('WEBUI_HOST')
        if args.port == 8000 and os.getenv('WEBUI_PORT'):
            args.port = int(os.getenv('WEBUI_PORT'))

    bot_clients_started = False
    if start_serve:
        if not prepare_webui_frontend_assets():
            logger.warning("前端靜態資源未就緒，繼續啟動 FastAPI 服務（Web 頁面可能不可用）")
        try:
            start_api_server(host=args.host, port=args.port, config=config)
            bot_clients_started = True
        except Exception as e:
            logger.error(f"啟動 FastAPI 服務失敗: {e}")

    if bot_clients_started:
        start_bot_stream_clients(config)

    # === 僅 Web 服務模式：不自動執行分析 ===
    if args.serve_only:
        logger.info("模式: 僅 Web 服務")
        logger.info(f"Web 服務運行中: http://{args.host}:{args.port}")
        logger.info("通過 /api/v1/analysis/stock/{code} 接口觸發分析")
        logger.info(f"API 文檔: http://{args.host}:{args.port}/docs")
        logger.info("按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n用戶中斷，程序退出")
        return 0

    try:
        # 模式0: 回測
        if getattr(args, 'backtest', False):
            logger.info("模式: 回測")
            from src.services.backtest_service import BacktestService

            service = BacktestService()
            stats = service.run_backtest(
                code=getattr(args, 'backtest_code', None),
                force=getattr(args, 'backtest_force', False),
                eval_window_days=getattr(args, 'backtest_days', None),
            )
            logger.info(
                f"回測完成: processed={stats.get('processed')} saved={stats.get('saved')} "
                f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
            )
            return 0

        # 模式1: 僅大盤復盤
        if args.market_review:
            from src.analyzer import GeminiAnalyzer
            from src.core.market_review import run_market_review
            from src.notification import NotificationService
            from src.search_service import SearchService

            # Issue #373: Trading day check for market-review-only mode.
            # Do NOT use _compute_trading_day_filter here: that helper checks
            # config.market_review_enabled, which would wrongly block an
            # explicit --market-review invocation when the flag is disabled.
            effective_region = None
            if not getattr(args, 'force_run', False) and getattr(config, 'trading_day_check_enabled', True):
                from src.core.trading_calendar import get_open_markets_today, compute_effective_region as _compute_region
                open_markets = get_open_markets_today()
                effective_region = _compute_region(
                    getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
                )
                if effective_region == '':
                    logger.info("今日大盤復盤相關市場均為非交易日，跳過執行。可使用 --force-run 強制執行。")
                    return 0

            logger.info("模式: 僅大盤復盤")
            notifier = NotificationService()

            # 初始化搜索服務和分析器（如果有配置）
            search_service = None
            analyzer = None

            if config.bocha_api_keys or config.tavily_api_keys or config.brave_api_keys or config.serpapi_keys:
                search_service = SearchService(
                    bocha_keys=config.bocha_api_keys,
                    tavily_keys=config.tavily_api_keys,
                    brave_keys=config.brave_api_keys,
                    serpapi_keys=config.serpapi_keys,
                    news_max_age_days=config.news_max_age_days,
                )

            if config.gemini_api_key or config.openai_api_key:
                analyzer = GeminiAnalyzer(api_key=config.gemini_api_key)
                if not analyzer.is_available():
                    logger.warning("AI 分析器初始化後不可用，請檢查 API Key 配置")
                    analyzer = None
            else:
                logger.warning("未檢測到 API Key (Gemini/OpenAI)，將僅使用模板生成報告")

            run_market_review(
                notifier=notifier,
                analyzer=analyzer,
                search_service=search_service,
                send_notification=not args.no_notify,
                override_region=effective_region,
            )
            return 0

        # 模式2: 定時任務模式
        if args.schedule or config.schedule_enabled:
            logger.info("模式: 定時任務")
            logger.info(f"每日執行時間: {config.schedule_time}")

            # Determine whether to run immediately:
            # Command line arg --no-run-immediately overrides config if present.
            # Otherwise use config (defaults to True).
            should_run_immediately = config.schedule_run_immediately
            if getattr(args, 'no_run_immediately', False):
                should_run_immediately = False

            logger.info(f"啟動時立即執行: {should_run_immediately}")

            from src.scheduler import run_with_schedule

            def scheduled_task():
                run_full_analysis(config, args, stock_codes)

            run_with_schedule(
                task=scheduled_task,
                schedule_time=config.schedule_time,
                run_immediately=should_run_immediately
            )
            return 0

        # 模式3: 正常單次運行
        if config.run_immediately:
            run_full_analysis(config, args, stock_codes)
        else:
            logger.info("配置為不立即運行分析 (RUN_IMMEDIATELY=false)")

        logger.info("\n程序執行完成")

        # 如果啟用了服務且是非定時任務模式，保持程序運行
        keep_running = start_serve and not (args.schedule or config.schedule_enabled)
        if keep_running:
            logger.info("API 服務運行中 (按 Ctrl+C 退出)...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        return 0

    except KeyboardInterrupt:
        logger.info("\n用戶中斷，程序退出")
        return 130

    except Exception as e:
        logger.exception(f"程序執行失敗: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
