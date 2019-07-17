british = readtable('Dummy_british_pats.csv', 'ReadRowNames', true);

eng_ind = ismember(british.FirstLanguages, {'English' 'English(NL)' 'english'});
british = british(eng_ind, :);

% Load pretrained growth models
load('lesion_growth_mdls_dummy.mat', 'growth_models');

% Find hold out set as ploras Patients who have scan and CAT that are not
% contemperaneous
hold_out_ind = (abs(british.DaysBetweenCATAndScan)>364);
hold_out = british(hold_out_ind, :);
british = british(~hold_out_ind,:);

% remove any patients who are repeated across ploras and hold out
hold_out_pats = hold_out.Properties.RowNames;
british = british(~startsWith(british.Properties.RowNames, hold_out_pats), :);

hold_out = hold_out(~isnan(hold_out.(task)), :);
rep_ind = ismember(british.FirstOrRepeat,{'Repeat'});
nan_spk = isnan(british.(task));
ind_voxvol = british.TotalVolumeVoxels ~= 0;
over_a_week = abs(british.DaysBetweenCATAndScan)>7;
british = british(~rep_ind& ~nan_spk & ind_voxvol & ~over_a_week, :);


CV_behavior_model = fitrgp(british, [task ' ~ ' sprintf('%s+',preds{1:end-1}),preds{end}], 'Kfold', 10, 'Standardize', true);
y_hat= CV_behavior_model.kfoldPredict;

ploras_pred_err = british.(task) - y_hat;

% Evaluate on hold out set
behavior_model = fitrgp(british, [task ' ~ ' sprintf('%s+',preds{1:end-1}),preds{end}], 'Standardize', true);
y_hat_hold_out = behavior_model.predict(hold_out);

hold_out_pred_err = hold_out.(task) - y_hat_hold_out;

%% Forward Project Hold Out Set
% hold_out.MonthsBetweenStrokeAndCAT = years(hold_out.CATDate - hold_out.DateOfStroke)*12;
hold_out.MonthsBetweenStrokeAndCAT_log = log(hold_out.MonthsBetweenStrokeAndCAT+1);
hold_out.MonthsBetweenStrokeAndScan_log = log(hold_out.MonthsBetweenStrokeAndScan+1);
resp_labs = [pc_labs{:} {'TotalVolumeVoxels'}];
pred_labs_growth = [pc_labs{:}, {'TotalVolumeVoxels', 'MonthsBetweenStrokeAndScan_log', 'MonthsBetweenStrokeAndCAT_log'}];
hold_out_proj = hold_out;

model_names = fieldnames(growth_models);
for m = 1:numel(resp_labs)
    hold_out_proj{:, resp_labs{m}} = growth_models.(model_names{m}).predict(hold_out{:, pred_labs_growth});
end

% Predict Behavior
y_hat_hold_out_proj = behavior_model.predict(hold_out_proj);
hold_out_pred_err_proj = hold_out.(task) - y_hat_hold_out_proj;

figure(2); clf; scatter(abs(hold_out_pred_err), abs(hold_out_pred_err_proj), 20, 'd', 'filled'); hold on;
plot([0,40],[0,40])
xlabel('Absolute Prediction Error Before Lesion Growth');
ylabel('Absolute Prediction Error After Lesion Growth');set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');

% Before and After Projection
figure(4); clf; scatter(hold_out.(task), y_hat_hold_out, 20, 'd', 'filled');
hold on; scatter(hold_out.(task), y_hat_hold_out_proj, 20, 'd', 'filled');
plot([0,70], [0,70])
for i = 1:height(hold_out)
    plot([hold_out{i, task}, hold_out{i, task}], [y_hat_hold_out(i), y_hat_hold_out_proj(i)], 'r--');
end
legend('Before Growth', 'After Growth', 'Location', 'best');
xlabel('Actual Score')
ylabel('Predicted Score')
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman')

% Significance test
[p,h,stats] = signrank(abs(hold_out_pred_err), abs(hold_out_pred_err_proj), 'tail','right')



