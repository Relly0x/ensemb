import torch
import onnx
import numpy as np
import logging
import os


class ONNXExporter:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('onnx_exporter')

    def export_model(self, model, path, dummy_batch=None):
        """
        Export PyTorch model to ONNX format

        Parameters:
        - model: Trained PyTorch model
        - path: Path to save the ONNX model
        - dummy_batch: Sample batch for tracing (required for our model)

        Returns:
        - True if successful, False otherwise
        """
        try:
            # Create export directory
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Set model to evaluation mode
            model.eval()

            if dummy_batch is None:
                # Create dummy inputs if not provided
                dummy_batch = self._create_dummy_batch()

            # Prepare inputs for ONNX export
            dummy_inputs = self._prepare_onnx_inputs(dummy_batch)

            # Get input names and shapes
            input_names = ['past_data', 'future_data', 'static_data']
            output_names = ['predictions']

            # Dynamic axes for variable batch size
            dynamic_axes = {
                'past_data': {0: 'batch_size'},
                'future_data': {0: 'batch_size'},
                'static_data': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }

            self.logger.info(f"Exporting model to ONNX format...")
            self.logger.info(
                f"Input shapes: past={dummy_inputs[0].shape}, future={dummy_inputs[1].shape}, static={dummy_inputs[2].shape}")

            # Create a wrapper model that accepts individual tensors
            wrapper_model = ONNXModelWrapper(model)

            # Export the model
            torch.onnx.export(
                wrapper_model,  # model being run
                dummy_inputs,  # model input (tuple of tensors)
                path,  # where to save the model
                export_params=True,  # store the trained parameters
                opset_version=11,  # the ONNX version to export the model to
                do_constant_folding=True,  # optimization
                input_names=input_names,  # model's input names
                output_names=output_names,  # model's output names
                dynamic_axes=dynamic_axes  # variable length axes
            )

            # Verify the model
            try:
                onnx_model = onnx.load(path)
                onnx.checker.check_model(onnx_model)
                self.logger.info(f"Model successfully exported and validated: {path}")

                # Test the exported model
                self._test_onnx_model(path, dummy_inputs)

                return True

            except Exception as e:
                self.logger.error(f"ONNX model validation failed: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error exporting model to ONNX: {e}")
            return False

    def _create_dummy_batch(self):
        """Create dummy batch for ONNX export"""
        # Use config to determine dimensions
        past_seq_len = self.config['model'].get('past_sequence_length', 120)
        forecast_horizon = self.config['model'].get('forecast_horizon', 12)

        # Create dummy data with reasonable dimensions
        dummy_batch = {
            'past': torch.randn(1, past_seq_len, 27),  # 27 features from normalizer
            'future': torch.randn(1, forecast_horizon, 26),  # 26 features (excluding close)
            'static': torch.randn(1, 1),  # 1 static feature
            'target': torch.randn(1, forecast_horizon)  # target (not used in export)
        }

        return dummy_batch

    def _prepare_onnx_inputs(self, batch):
        """Prepare inputs for ONNX export"""
        # Extract tensors from batch dictionary
        past_data = batch['past']
        future_data = batch['future']
        static_data = batch['static']

        return (past_data, future_data, static_data)

    def _test_onnx_model(self, onnx_path, dummy_inputs):
        """Test the exported ONNX model"""
        try:
            import onnxruntime as ort

            # Create inference session
            ort_session = ort.InferenceSession(onnx_path)

            # Get input names
            input_names = [input.name for input in ort_session.get_inputs()]

            # Prepare inputs for ONNX Runtime
            ort_inputs = {}
            for i, name in enumerate(input_names):
                ort_inputs[name] = dummy_inputs[i].numpy()

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)

            self.logger.info(f"ONNX model test successful. Output shape: {ort_outputs[0].shape}")

        except ImportError:
            self.logger.warning("ONNX Runtime not available for testing. Install with: pip install onnxruntime")
        except Exception as e:
            self.logger.error(f"ONNX model test failed: {e}")


class ONNXModelWrapper(torch.nn.Module):
    """
    Wrapper to make the model compatible with ONNX export
    Converts individual tensor inputs back to the dictionary format expected by the model
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, past_data, future_data, static_data):
        """
        Forward pass that converts individual tensors to dictionary format
        """
        # Recreate the batch dictionary expected by the model
        batch = {
            'past': past_data,
            'future': future_data,
            'static': static_data
        }

        # Call the original model
        return self.model(batch)