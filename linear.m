% Load the dataset
load('data3.mat');

% Extract features (first two columns) and labels (last column)
X = data(:, 1:end-1); % Features matrix (200x2)
y = data(:, end); % Labels vector (200x1)

% Initialize parameters
[num_samples, num_features] = size(X); % Get the number of samples and features
w = randn(num_features, 1); % Initialize weights randomly
b = randn(); % Initialize bias randomly
r = 0.1; % Define the learning rate (step size)
max_iter = 1000; % Set the maximum number of iterations
tol = 1e-5; % Define the convergence tolerance

% Arrays to keep track of error progression
class_error = zeros(max_iter, 1);
perc_loss = zeros(max_iter, 1);

for iteration = 1:max_iter
    % Calculate predictions
    pred = X * w + b;
    misclassified_indices = find(y .* pred <= 0); % Identify misclassified samples

    % Update weights and bias
    weight_gradient = -sum(bsxfun(@times, y(misclassified_indices), X(misclassified_indices, :)), 1)';
    bias_gradient = -sum(y(misclassified_indices));
    
    w = w - r * weight_gradient / num_samples;
    b = b - r * bias_gradient / num_samples;

    % Compute binary classification error and perceptron loss
    class_error(iteration) = sum(y .* pred <= 0) / num_samples;
    perc_loss(iteration) = sum(-y .* pred .* (y .* pred <= 0)) / num_samples;

    % Check for convergence
    if norm(weight_gradient) < tol && abs(bias_gradient) < tol
        fprintf('Converged after %d iterations.\n', iteration);
        break;
    end
end

% Plotting the decision boundary
figure;
hold on;
scatter(X(y == 1, 1), X(y == 1, 2), 'r'); % Scatter plot for class 1
scatter(X(y == -1, 1), X(y == -1, 2), 'b'); % Scatter plot for class -1
x_range = linspace(min(X(:, 1)), max(X(:, 1)), 100); % Range for x-axis
y_range = -(w(1) * x_range + b) / w(2); % Calculate y values for decision boundary
plot(x_range, y_range, 'k-', 'LineWidth', 2); % Plot the decision boundary
xlabel('Feature 1'); % Label for x-axis
ylabel('Feature 2'); % Label for y-axis
title('Linear Decision Boundary'); % Title for the plot
hold off;

% Plotting the evolution of error
figure;
subplot(2, 1, 1);
plot(1:iteration, class_error(1:iteration), 'r'); % Plot binary classification error
title('Binary Classification Error'); % Title for the error plot
xlabel('Iterations'); % Label for x-axis
ylabel('Error'); % Label for y-axis

subplot(2, 1, 2);
plot(1:iteration, perc_loss(1:iteration), 'b'); % Plot perceptron loss
title('Perceptron Loss'); % Title for the loss plot
xlabel('Iterations'); % Label for x-axis
ylabel('Error'); % Label for y-axis
