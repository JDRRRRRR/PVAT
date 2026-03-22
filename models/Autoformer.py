import torch
import torch.nn as nn
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Embed import DataEmbedding_wo_pos

class Model(nn.Module):

    def __init__(self,
                 task_name='long_term_forecast',
                 seq_len=96,
                 label_len=48,
                 pred_len=96,
                 output_attention=False,
                 moving_avg=25,
                 enc_in=7,
                 dec_in=7,
                 d_model=512,
                 embed='timeF',
                 freq='h',
                 dropout=0.1,
                 factor=3,
                 n_heads=8,
                 d_ff=2048,
                 activation='gelu',
                 e_layers=2,
                 d_layers=1,
                 c_out=7,
                 num_class=10):
        super(Model, self).__init__()

        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Encoder embedding: embed encoder input (without positional encoding)
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)

        # Decoder embedding: embed decoder input (without positional encoding)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # Stack of encoder layers with AutoCorrelation attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # AutoCorrelation layer for attention
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=False,  # No masking for encoder
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # Decoder configuration depends on task type
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Forecasting decoder with trend extrapolation
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        # Self-attention on decoder input (with masking)
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                mask_flag=True,  # Causal masking for decoder
                                factor=factor,
                                attention_dropout=dropout,
                                output_attention=False
                            ),
                            d_model, n_heads
                        ),
                        # Cross-attention between decoder and encoder
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                mask_flag=False,  # No masking for cross-attention
                                factor=factor,
                                attention_dropout=dropout,
                                output_attention=False
                            ),
                            d_model, n_heads
                        ),
                        d_model,
                        c_out,
                        d_ff,
                        moving_avg=moving_avg,
                        dropout=dropout,
                        activation=activation,
                    ) for _ in range(d_layers)
                ],
                norm_layer=my_Layernorm(d_model),
                projection=nn.Linear(d_model, c_out, bias=True)
            )

        if self.task_name == 'imputation':
            # Projection for imputation task
            self.projection = nn.Linear(d_model, c_out, bias=True)

        if self.task_name == 'anomaly_detection':
            # Projection for anomaly detection task
            self.projection = nn.Linear(d_model, c_out, bias=True)

        if self.task_name == 'classification':
            # Classification head
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(dropout)
            # Flatten and project to number of classes
            self.projection = nn.Linear(d_model * seq_len, num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        # Create zero tensor for seasonal component padding
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)

        # Decompose encoder input into seasonal and trend
        seasonal_init, trend_init = self.decomp(x_enc)

        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)

        # Seasonal: concatenate last label_len steps with zeros for prediction
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        # Apply decoder layers with trend context
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out,
            x_mask=None,
            cross_mask=None,
            trend=trend_init
        )

        dec_out = trend_part + seasonal_part

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):

        # Embed encoder input
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project to output dimension
        dec_out = self.projection(enc_out)

        return dec_out

    def anomaly_detection(self, x_enc):

        # Embed encoder input (no time features for anomaly detection)
        enc_out = self.enc_embedding(x_enc, None)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project to output dimension
        dec_out = self.projection(enc_out)

        return dec_out

    def classification(self, x_enc, x_mark_enc):

        # Embed encoder input (no time features for classification)
        enc_out = self.enc_embedding(x_enc, None)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Apply activation function
        output = self.act(enc_out)

        # Apply dropout
        output = self.dropout(output)

        # Apply time feature weighting if provided
        output = output * x_mark_enc.unsqueeze(-1) if x_mark_enc is not None else output

        # Flatten: [batch, seq_len, d_model] -> [batch, seq_len * d_model]
        output = output.reshape(output.shape[0], -1)

        # Project to class logits
        output = self.projection(output)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        # Route to appropriate task handler
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]

        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out

        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out

        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out

        return None
