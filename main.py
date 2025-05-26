# main.py - Interactive Trading Bot Controller

import argparse
import json
import os
import logging
import sys
from datetime import datetime


def setup_logger(level="INFO", log_file=None):
    """Simple logger setup"""
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger('tft_trading_bot')


def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return None


def validate_config(config):
    """Simple configuration validation"""
    required_sections = ['data', 'model', 'strategy', 'execution']
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required section: {section}")
            return False
    return True


def show_menu():
    """Display interactive menu"""
    print("\n" + "=" * 60)
    print("🤖 ENHANCED TFT TRADING BOT CONTROLLER")
    print("=" * 60)
    print("📊 Choose what you want to do:")
    print()
    print("1️⃣  Train Single TFT Model")
    print("2️⃣  Train Complete 5-Model Ensemble (TFT+LSTM+XGBoost+Transformer+CNN)")
    print("3️⃣  Quick 3-Model Ensemble (TFT+LSTM+XGBoost)")
    print("4️⃣  Run Live Trading (Single Model)")
    print("5️⃣  Run Live Trading (Ensemble)")
    print("6️⃣  Backtest Single Model")
    print("7️⃣  Backtest Ensemble")
    print("8️⃣  Export Models to ONNX")
    print("9️⃣  View Training Status/Logs")
    print("0️⃣  Exit")
    print()
    print("=" * 60)


def get_user_choice():
    """Get and validate user choice"""
    while True:
        try:
            choice = input("👉 Enter your choice (0-9): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return choice
            else:
                print("❌ Invalid choice! Please enter a number between 0-9")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)


def choose_config():
    """Let user choose configuration"""
    print("\n📁 Choose configuration:")
    print("1. Single Model Config (config/config.json)")
    print("2. Ensemble Config (config/ensemble_config.json)")
    print("3. Custom config path")

    config_choice = input("👉 Enter choice (1-3): ").strip()

    if config_choice == "1":
        return "config/config.json"
    elif config_choice == "2":
        return "config/ensemble_config.json"
    elif config_choice == "3":
        return input("👉 Enter config path: ").strip()
    else:
        print("❌ Invalid choice, using default config/config.json")
        return "config/config.json"


def load_model(model_path):
    """Load a trained model"""
    import torch
    from models.tft.model import TemporalFusionTransformer, SimpleTFT

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract model config from checkpoint if available
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        # Use default config
        model_config = {
            'hidden_size': 64,
            'attention_heads': 4,
            'lstm_layers': 2,
            'dropout': 0.1,
            'past_sequence_length': 120,
            'forecast_horizon': 12,
            'quantiles': [0.1, 0.5, 0.9],
            'static_input_dim': 1,
            'past_input_dim': 50,
            'future_input_dim': 10
        }

    # Try to create TFT model first, fallback to SimpleTFT if needed
    try:
        state_dict_keys = checkpoint.get('model_state_dict', {}).keys()
        has_tft_layers = any('attention' in key or 'variable_selection' in key for key in state_dict_keys)

        if has_tft_layers:
            model = TemporalFusionTransformer(model_config)
        else:
            model = SimpleTFT(model_config)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        model.eval()
        return model

    except Exception as e:
        print(f"❌ Failed to load as TFT model: {e}")
        print("🔄 Falling back to SimpleTFT...")

        model = SimpleTFT(model_config)
        if 'model_state_dict' in checkpoint:
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        model.eval()
        return model


def train_single_model(config):
    """Train single TFT model"""
    logger = logging.getLogger('single_model_trainer')

    try:
        print("\n🚀 Starting Single TFT Model Training...")
        print("📊 This will take about 15-20 minutes")

        # Import modules
        from data.collectors.oanda_collector import OandaDataCollector
        from data.processors.normalizer import DataNormalizer
        from models.tft.model import TemporalFusionTransformer
        from pipelines.training.trainer import TFTTrainer
        from data.dataset import create_datasets

        # Collect and process data
        logger.info("📊 Collecting data from OANDA...")
        collector = OandaDataCollector(config)
        data = collector.collect_training_data()

        if not data:
            print("❌ Failed to collect training data!")
            return False

        logger.info("🔧 Processing and normalizing data...")
        processor = DataNormalizer(config)
        processed_data = processor.process(data)

        logger.info("📦 Creating datasets...")
        train_loader, val_loader, test_loader = create_datasets(processed_data, config)

        logger.info("🧠 Initializing TFT model...")
        model = TemporalFusionTransformer(config['model'])

        # Initialize model with dummy batch
        dummy_batch = next(iter(train_loader))
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for key in dummy_batch:
            if isinstance(dummy_batch[key], torch.Tensor):
                dummy_batch[key] = dummy_batch[key].to(device)

        with torch.no_grad():
            try:
                _ = model(dummy_batch)
                logger.info("✅ Model initialized successfully")
            except Exception as e:
                logger.error(f"❌ Model initialization failed: {e}")
                from models.tft.model import SimpleTFT
                model = SimpleTFT(config['model']).to(device)
                logger.info("🔄 Using SimpleTFT instead")

        # Train model
        logger.info("🎯 Starting training...")
        trainer = TFTTrainer(config, model, train_loader, val_loader, test_loader)
        test_metrics = trainer.train()

        print(f"\n✅ Training completed!")
        print(f"📊 Test metrics: {test_metrics}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_ensemble(config, model_list=None):
    """Train ensemble models"""
    logger = logging.getLogger('ensemble_trainer')

    if model_list is None:
        model_list = ['tft', 'lstm', 'xgboost', 'transformer', 'cnn']

    try:
        print(f"\n🚀 Starting Ensemble Training with {len(model_list)} models...")
        print(f"📊 Models: {', '.join(model_list).upper()}")
        print(f"⏱️ Estimated time: {len(model_list) * 12} minutes")

        # Ensure ensemble is enabled in config
        config['ensemble']['enabled'] = True
        config['ensemble']['enabled_models'] = model_list

        # Import modules
        from pipelines.training.ensemble_trainer import EnsembleTrainer
        from data.collectors.oanda_collector import OandaDataCollector
        from data.processors.normalizer import DataNormalizer
        from data.dataset import create_datasets

        # Collect and process data
        logger.info("📊 Collecting training data...")
        collector = OandaDataCollector(config)
        data = collector.collect_training_data()

        if not data:
            print("❌ Failed to collect training data!")
            return False

        logger.info("🔧 Processing data...")
        processor = DataNormalizer(config)
        processed_data = processor.process(data)

        logger.info("📦 Creating datasets...")
        train_loader, val_loader, test_loader = create_datasets(processed_data, config)

        # Train ensemble
        logger.info("🎯 Starting ensemble training...")
        ensemble_trainer = EnsembleTrainer(config, train_loader, val_loader, test_loader)
        results = ensemble_trainer.train_all_models()

        # Show results
        print("\n" + "=" * 60)
        print("🏆 ENSEMBLE TRAINING RESULTS:")
        print("=" * 60)

        successful_models = []
        failed_models = []

        for model_name, result in results.items():
            if result.get('success'):
                val_loss = result.get('best_val_loss') or result.get('val_loss', 'N/A')
                print(f"✅ {model_name.upper()}: Success (Val Loss: {val_loss})")
                successful_models.append(model_name)
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ {model_name.upper()}: Failed - {error}")
                failed_models.append(model_name)

        print("=" * 60)
        print(f"🎉 Summary: {len(successful_models)}/{len(model_list)} models trained successfully")
        print(f"🏆 Working models: {', '.join(successful_models)}")

        if failed_models:
            print(f"⚠️ Failed models: {', '.join(failed_models)}")

        return len(successful_models) > 0

    except Exception as e:
        print(f"❌ Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_live_trading(config, use_ensemble=False):
    """Run live trading"""
    try:
        if use_ensemble:
            print("\n🚀 Starting Live Trading with Ensemble...")

            from pipelines.inference.ensemble_inference_engine import EnsembleInferenceEngine
            from strategy.enhanced_strategy_factory import create_enhanced_strategy
            from execution.execution_engine import ExecutionEngine

            # Initialize ensemble
            ensemble_engine = EnsembleInferenceEngine(config)
            strategy = create_enhanced_strategy(config, ensemble_engine=ensemble_engine)
            engine = ExecutionEngine(config, ensemble_engine, strategy)
        else:
            print("\n🚀 Starting Live Trading with Single Model...")

            from strategy.strategy_factory import create_strategy
            from execution.execution_engine import ExecutionEngine

            # Load model
            model_path = config.get('export', {}).get('model_path', 'models/checkpoints/best_model.pt')
            model = load_model(model_path)

            strategy = create_strategy(config)
            engine = ExecutionEngine(config, model, strategy)

        print("✅ Starting live trading...")
        print("⚠️ Press Ctrl+C to stop safely")
        engine.start_live_trading()

    except Exception as e:
        print(f"❌ Live trading failed: {e}")
        return False


def run_backtest(config, use_ensemble=False):
    """Run backtest"""
    try:
        if use_ensemble:
            print("\n📈 Starting Ensemble Backtest...")

            from pipelines.inference.ensemble_inference_engine import EnsembleInferenceEngine
            from strategy.enhanced_strategy_factory import create_enhanced_strategy
            from execution.execution_engine import ExecutionEngine

            ensemble_engine = EnsembleInferenceEngine(config)
            strategy = create_enhanced_strategy(config, ensemble_engine=ensemble_engine)
            engine = ExecutionEngine(config, ensemble_engine, strategy)
        else:
            print("\n📈 Starting Single Model Backtest...")

            from strategy.strategy_factory import create_strategy
            from execution.execution_engine import ExecutionEngine

            model_path = config.get('export', {}).get('model_path', 'models/checkpoints/best_model.pt')
            model = load_model(model_path)

            strategy = create_strategy(config)
            engine = ExecutionEngine(config, model, strategy)

        # Run backtest
        results = engine.backtest()

        print("\n📊 BACKTEST RESULTS:")
        print("=" * 40)
        for key, value in results.items():
            if key != 'equity_curve':
                print(f"{key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return False


def export_models(config):
    """Export models to ONNX"""
    try:
        print("\n📦 Exporting Models to ONNX...")

        from pipelines.export.onnx_exporter import ONNXExporter

        model_path = config.get('export', {}).get('model_path', 'models/checkpoints/best_model.pt')
        model = load_model(model_path)

        export_path = os.path.join(config.get('export', {}).get('export_dir', 'exported_models'), 'model.onnx')
        exporter = ONNXExporter(config)

        # Create dummy batch for export
        from data.collectors.oanda_collector import OandaDataCollector
        from data.processors.normalizer import DataNormalizer
        from data.dataset import create_datasets

        collector = OandaDataCollector(config)
        data = collector.collect_training_data()
        processor = DataNormalizer(config)
        processed_data = processor.process(data)
        train_loader, _, _ = create_datasets(processed_data, config)
        dummy_batch = next(iter(train_loader))

        success = exporter.export_model(model, export_path, dummy_batch)

        if success:
            print(f"✅ Model exported to {export_path}")
        else:
            print("❌ Export failed")

        return success

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def view_logs():
    """View training logs and status"""
    print("\n📋 TRAINING STATUS & LOGS:")
    print("=" * 50)

    # Check for model files
    model_dir = "models/checkpoints"
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        if files:
            print("📁 Available Models:")
            for file in files:
                if file.endswith('.pt') or file.endswith('.pkl'):
                    file_path = os.path.join(model_dir, file)
                    size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    print(f"   📄 {file} ({size:.1f}MB) - {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("📁 No trained models found")
    else:
        print("📁 Models directory not found")

    print()

    # Check for log files
    log_dir = "logs"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if log_files:
            print("📋 Recent Log Files:")
            for log_file in sorted(log_files)[-3:]:  # Show last 3
                log_path = os.path.join(log_dir, log_file)
                size = os.path.getsize(log_path) / 1024  # KB
                mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
                print(f"   📝 {log_file} ({size:.1f}KB) - {mtime.strftime('%Y-%m-%d %H:%M')}")

                # Show last few lines
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"      💬 Last line: {lines[-1].strip()}")
                except:
                    pass
        else:
            print("📋 No log files found")
    else:
        print("📋 Logs directory not found")


def main():
    """Main interactive function"""
    print("🤖 Enhanced TFT Trading Bot")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        show_menu()
        choice = get_user_choice()

        if choice == '0':
            print("\n👋 Thanks for using the Enhanced TFT Trading Bot!")
            print("💫 Happy trading!")
            break

        elif choice == '1':
            # Train Single TFT Model
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                train_single_model(config)

        elif choice == '2':
            # Train Complete 5-Model Ensemble
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                train_ensemble(config, ['tft', 'lstm', 'xgboost', 'transformer', 'cnn'])

        elif choice == '3':
            # Quick 3-Model Ensemble
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                train_ensemble(config, ['tft', 'lstm', 'xgboost'])

        elif choice == '4':
            # Live Trading (Single)
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                run_live_trading(config, use_ensemble=False)

        elif choice == '5':
            # Live Trading (Ensemble)
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                run_live_trading(config, use_ensemble=True)

        elif choice == '6':
            # Backtest Single
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                run_backtest(config, use_ensemble=False)

        elif choice == '7':
            # Backtest Ensemble
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                run_backtest(config, use_ensemble=True)

        elif choice == '8':
            # Export Models
            config_path = choose_config()
            config = load_config(config_path)
            if config and validate_config(config):
                logger = setup_logger(config.get('logging', {}).get('level', 'INFO'),
                                      config.get('logging', {}).get('file'))
                export_models(config)

        elif choice == '9':
            # View Logs
            view_logs()

        input("\n⏸️ Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()