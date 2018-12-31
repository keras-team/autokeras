from deepvoice3_pytorch import MultiSpeakerTTSModel, AttentionSeq2Seq


def deepvoice3(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
               downsample_step=1,
               n_speakers=1, speaker_embed_dim=16, padding_idx=0,
               dropout=(1 - 0.95), kernel_size=5,
               encoder_channels=128,
               decoder_channels=256,
               converter_channels=256,
               query_position_rate=1.0,
               key_position_rate=1.29,
               use_memory_mask=False,
               trainable_positional_encodings=False,
               force_monotonic_attention=True,
               use_decoder_state_for_postnet_input=True,
               max_positions=512,
               embedding_weight_std=0.1,
               speaker_embedding_weight_std=0.01,
               freeze_embedding=False,
               window_ahead=3,
               window_backward=1,
               key_projection=False,
               value_projection=False,
               ):
    """Build deepvoice3
    """
    from deepvoice3_pytorch.deepvoice3 import Encoder, Decoder, Converter

    time_upsampling = max(downsample_step // r, 1)

    # Seq2seq
    h = encoder_channels  # hidden dim (channels)
    k = kernel_size  # kernel size
    encoder = Encoder(
        n_vocab, embed_dim, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        embedding_weight_std=embedding_weight_std,
        # (channels, kernel_size, dilation)
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3)],
    )

    h = decoder_channels
    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        preattention=[(h, k, 1), (h, k, 3)],
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1)],
        attention=[True, False, False, False, True],
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask,
        window_ahead=window_ahead,
        window_backward=window_backward,
        key_projection=key_projection,
        value_projection=value_projection,
    )

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    # Post net
    if use_decoder_state_for_postnet_input:
        in_dim = h // r
    else:
        in_dim = mel_dim
    h = converter_channels
    converter = Converter(
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
        time_upsampling=time_upsampling,
        convolutions=[(h, k, 1), (h, k, 3), (2 * h, k, 1), (2 * h, k, 3)],
    )

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
        speaker_embedding_weight_std=speaker_embedding_weight_std,
        freeze_embedding=freeze_embedding)

    return model
