%% Read in data
british = readtable('Dummy_british_pats.csv', 'ReadRowNames', true);

% Task
task = 'SpkPicDescTOT';

% Ploras Exclusion Criteria
rep_ind = ismember(british.FirstOrRepeat,{'Repeat'});
eng_ind = ismember(british.FirstLanguages, {'English' 'English(NL)' 'english'});
nan_spk = isnan(british.(task));
ind_voxvol = (british.TotalVolumeVoxels ~= 0);
british = british(~rep_ind & eng_ind & ~nan_spk & ind_voxvol, :);

chile = readtable('Dummy_chile_pats.csv');

% Predictors
pc_labs = arrayfun(@(x) ['PC_' num2str(x)], 1:12, 'UniformOutput', false);
preds = {'TotalVolumeVoxels', 'MonthsBetweenStrokeAndCAT', 'AgeAtStroke', pc_labs{:}};

%% Perform cross validation on PLORAS
progressbar('CrossValdiation');
cv_partitions = arrayfun(@(x) cvpartition(height(british),'KFold',10), 1:10, 'UniformOutput', false);
y = british{:, task};
y_hat = zeros(numel(y), length(cv_partitions));

for c = 1:numel(cv_partitions)
    progressbar(c/numel(cv_partitions));
    CV_behavior_model = fitrgp(british, [task ' ~ ' sprintf('%s+',preds{1:end-1}),preds{end}], 'CVPartition', cv_partitions{c}, 'Standardize', true);
    % CV_behavior_model = fitrlinear(ploras{:, preds}, ploras.(task), , 'CVPartition', cv_partitions{c});
    y_hat(:, c) = CV_behavior_model.kfoldPredict;
end

y_hat = mean(y_hat,2);
fprintf('R_sqrd within ploras = %.2f\n', corr2(y, y_hat)^2);
PLORAS_pred_err = y - y_hat;

%% Plot PLORAS predictions
figure(2); clf; 
subplot(1,2,1);
scatter(british.(task), y_hat, 15, 'd', 'filled');
hold on
plot([min([british.(task)', y_hat']), max([british.(task)', y_hat'])], [min([british.(task)', y_hat']),max([british.(task)', y_hat'])])
% xlim([-10, 75])
% ylim([-10, 75])
xlabel('Actual Score')
ylabel('Predicted Score')
title('British');
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');

%% Test on Chile
behavior_model = fitrgp(british, [task ' ~ ' sprintf('%s+',preds{1:end-1}),preds{end}], 'Standardize', true);
% behavior_model = fitrlinear(ploras{:, preds}, ploras.(task));
chile_y_hat = behavior_model.predict(chile);
fprintf('R_sqrd chile before growth = %.2f\n', corr2(chile.(task), chile_y_hat)^2);
chile_pred_err = chile.(task) - chile_y_hat;

% Plot Predictions
subplot(1,2,2);
scatter(chile.(task), chile_y_hat, 15, 'd', 'filled');
xlabel('Actual Score')
ylabel('Predicted Score')
title('Chilean');
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman')
hold on
plot([min([british.(task)', y_hat']), max([british.(task)', y_hat'])], [min([british.(task)', y_hat']),max([british.(task)', y_hat'])])
% xlim([-10, 75])
% ylim([-10, 75])

%% Anova on prediction errors
g = ones(length(PLORAS_pred_err) + length(chile_pred_err), 1);
g(1:length(PLORAS_pred_err)) = 0;

scores = [british.(task)' chile.(task)']';
fitlm([g scores],  [PLORAS_pred_err', chile_pred_err'], 'CategoricalVar', logical([1, 0]), 'PredictorVars', {'Chile', 'Score'})
% Calculate Bayes factor
[~, ~, r] = regress([y_hat; chile_y_hat], [ones(length(scores),1) scores]);
fprintf('log(Bayes Factor) before lesion growth = %.3f\n', bayes_anova1(r, g+1));
% anovan([PLORAS_pred_err(ind)', chile_pred_err'], [g, scores], 'continuous', [2], 'varnames', {'Chile', 'Score'});

%% Learn and save models for lesion growth
Grow_Lesion_Model_Train

%% Forward project Chile data
load('Lesion_Growth_Models.mat', 'growth_models');
chile_forward = chile;
model_resp = [pc_labs, 'TotalVolumeVoxels'];
model_names = fieldnames(growth_models);
chile_labs = [pc_labs, 'TotalVolumeVoxels', 'MonthsBetweenStrokeAndScan_log', 'MonthsBetweenStrokeAndCAT_log'];
chile.MonthsBetweenStrokeAndScan_log = log(chile.MonthsBetweenStrokeAndScan+1);
chile.MonthsBetweenStrokeAndCAT_log = log(chile.MonthsBetweenStrokeAndCAT+1);
% Log transform time post stroke
for m = 1:numel(model_resp)
    model = growth_models.(model_names{m});
    chile_forward.(model_resp{m}) = model.predict(chile{:, chile_labs});
end

%% Test on Chile forward
chile_y_hat_proj = behavior_model.predict(chile_forward);

fprintf('R chile after growth = %.2f\n', corr2(chile_forward.(task), chile_y_hat_proj));
chile_pred_err_proj = chile_forward.(task) - chile_y_hat_proj;

%% Anova on prediction errors
g = ones(length(PLORAS_pred_err) + length(chile_pred_err_proj), 1);
g(1:length(PLORAS_pred_err)) = 0;

scores = [british.(task)' chile_forward.(task)']';
fitlm([g scores],  [PLORAS_pred_err', chile_pred_err_proj'], 'CategoricalVar', logical([1, 0]), 'PredictorVars', {'Group', 'Score'})
[~, ~, r] = regress([y_hat; chile_y_hat_proj], [ones(length(scores),1) scores]);
fprintf('log(Bayes Factor) after lesion growth = %.3f\n', bayes_anova1(r, g+1));
% anovan([PLORAS_pred_err', chile_pred_err'], [g, scores], 'continuous', [2], 'varnames', {'Chile', 'Score'});

%% Plot before and after manipulation
chile_y = chile.(task);
figure(4); clf; scatter(chile_y, chile_y_hat, 20, 'd', 'filled');
hold on; scatter(chile_y, chile_y_hat_proj, 20, 'd', 'filled');
plot([-10, max([british.(task)', y_hat'])], [-10,max([british.(task)', y_hat'])], 'Color',[217,83,25]/255)
xlim([-10, max([chile.(task)', chile_y_hat_proj'])])
ylim([-10, max(chile.(task))])
for i = 1:numel(chile_y)
plot([chile_y(i), chile_y(i)], [chile_y_hat(i), chile_y_hat_proj(i)], 'r--');
end
legend('Before Growth', 'After Growth', 'Location', 'best');
xlabel('Actual Score')
ylabel('Predicted Score')
% title('Chile PLORAS');
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman')

figure(5); clf;
boxplot([chile_pred_err, chile_pred_err_proj]);
xticklabels({'Before Growth', 'After Growth'});
ylabel('Prediction Error');

[p,h,stats] = signrank(abs(chile_pred_err), abs(chile_pred_err_proj), 'tail','right')


