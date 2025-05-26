import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit for Temporal Fusion Transformer
    GLU(x, W, U, b, c) = σ(xW + b) ⊙ (xU + c)
    where σ is the sigmoid activation and ⊙ is element-wise multiplication
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        # Project input to twice the size (for gating mechanism)
        x = self.linear(x)

        # Split the tensor into two equal parts along the last dimension
        gates, values = torch.chunk(x, 2, dim=-1)

        # Apply sigmoid activation to the gates
        gates = torch.sigmoid(gates)

        # Element-wise multiplication
        return gates * values


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) as described in the TFT paper
    GRN includes:
    1. ELU activation
    2. Layer normalization
    3. Gating mechanism
    4. Residual connection
    5. Dropout
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if output_dim != input_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None

        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Gating layer
        self.gate = GatedLinearUnit(input_dim, output_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Compute skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        # First layer with ELU activation
        hidden = F.elu(self.fc1(x))

        # Second layer
        hidden = self.fc2(hidden)

        # Apply gating mechanism
        gated_hidden = self.gate(x)

        # Combine with skip connection
        outputs = self.layer_norm(gated_hidden + skip)

        # Apply dropout
        outputs = self.dropout(outputs)

        return outputs


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) as described in the TFT paper
    VSN uses GRNs to provide variable selection and feature processing
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Variable selection GRN
        self.selection_grn = GatedResidualNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,  # Output one weight per variable
            dropout=dropout
        )

        # Variable processing GRNs, one per input variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=1,  # Each variable is processed independently
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout
            ) for _ in range(input_dim)
        ])

    def forward(self, x):
        # x shape: [batch_size, input_dim] or [batch_size, seq_len, input_dim]

        # Check if input is 3D (with sequence dimension)
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            flatten_dims = True
            # Reshape to [batch_size * seq_len, input_dim]
            x_reshaped = x.reshape(-1, self.input_dim)
        else:
            batch_size = x.shape[0]
            flatten_dims = False
            x_reshaped = x

        # Calculate variable selection weights
        selection_weights = self.selection_grn(x_reshaped)
        selection_weights = torch.softmax(selection_weights, dim=-1)

        # Process each variable independently
        processed_vars = []
        for i, grn in enumerate(self.variable_grns):
            # Extract the i-th variable and process it
            var = x_reshaped[:, i:i + 1]
            processed_var = grn(var)
            processed_vars.append(processed_var)

        # Stack processed variables
        processed_x = torch.stack(processed_vars, dim=-2)

        # Apply variable selection weights
        selection_weights = selection_weights.unsqueeze(-1)
        outputs = torch.sum(selection_weights * processed_x, dim=-2)

        # Reshape back to original dimensions if input was 3D
        if flatten_dims:
            outputs = outputs.reshape(batch_size, seq_len, self.output_dim)

        return outputs