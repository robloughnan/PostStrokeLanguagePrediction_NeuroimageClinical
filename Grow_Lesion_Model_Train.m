% Read in Data
repeated_scans = readtable('Dummy_repeated_scans.csv', 'ReadRowNames', true);

%% Derive Measures and define models
repeated_scans.MonthsStroketoT2 = repeated_scans.MonthsStroketoT1 + repeated_scans.MonthsT1toT2;
repeated_scans.MonthsStroketoT2_log = log(repeated_scans.MonthsStroketoT2+1);
repeated_scans.MonthsStroketoT1_log = log(repeated_scans.MonthsStroketoT1+1);
pc_labs = arrayfun(@(x) ['PC_' num2str(x)], 1:12, 'UniformOutput', false);
pred_labs = [cellfun(@(x) [x '_a'], pc_labs, 'UniformOutput', false), {'VolTa', 'MonthsStroketoT1_log', 'MonthsStroketoT2_log'}];
pred_labs_gpr = [cellfun(@(x) [x '_a'], pc_labs, 'UniformOutput', false), {'VolTa', 'MonthsStroketoT1', 'MonthsStroketoT2'}];
resp_labs = [cellfun(@(x) [x '_b'], pc_labs, 'UniformOutput', false) {'VolTb'}];


%% Train Models
growth_models = struct();
for i = 1:numel(resp_labs)
    growth_models.(resp_labs{i}) = fitlm(repeated_scans{:, pred_labs}, repeated_scans.(resp_labs{i}), 'PredictorVars', pred_labs);
end

save('lesion_growth_mdls_dummy.mat', 'growth_models');
