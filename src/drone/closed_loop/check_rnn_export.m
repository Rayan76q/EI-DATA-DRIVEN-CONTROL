function report = check_rnn_export(export_dir)
%CHECK_RNN_EXPORT Compare recurrence and physical normalization with Python.
P = load(fullfile(export_dir, 'rnn.mat'));
hidden = zeros(double(P.hidden_size), double(P.n_layers));
maximum_error = 0;
for k = 1:size(P.validation_inputs, 1)
    x = (double(P.validation_inputs(k, :))' - P.input_mean(:)) ./ P.input_scale(:);
    for layer = 1:double(P.n_layers)
        input_term = double(P.(sprintf('Wih%d', layer))) * x + P.(sprintf('bih%d', layer));
        state_term = double(P.(sprintf('Whh%d', layer))) * hidden(:, layer) + P.(sprintf('bhh%d', layer));
        x = tanh(input_term + state_term);
        hidden(:, layer) = x;
    end
    output = (double(P.head_weight) * x + P.head_bias) .* P.output_scale(:) + P.output_mean(:);
    expected = P.validation_outputs(k, :)';
    maximum_error = max(maximum_error, max(abs(output - expected)));
    assert(all(abs(output - expected) < 2e-4 + 2e-5 * abs(expected)), ...
        'RNN prediction disagrees with Python at sample %d.', k);
end
report = struct('status', 'passed', 'samples', size(P.validation_inputs, 1), ...
    'max_prediction_error', maximum_error);
fprintf('MATLAB RNN check passed: %d samples, max error %.3g.\n', ...
    report.samples, maximum_error);
end
