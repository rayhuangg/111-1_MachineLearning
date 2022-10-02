clc; clear;

% 讀取資料
X = load('02HW1_Xtrain');
Y = load('02HW1_Ytrain');


for n=1:3
    subplot(1, 3, n)
    plot(X, Y, "o");
    hold on;
    
    [p, S] = polyfit(X ,Y, n); % 線性擬合
    y1 = polyval(p, X);        % 計算預測值
    T = table(X, Y, y1, Y-y1, 'VariableNames', {'X','Y','Fit','FitError'})
    sum(Y-y1)                  % 計算實際X點誤差
    
    x1 = linspace(0, 1, 20);   % 建立新X軸以便做圖
    y1 = polyval(p, x1);       % 繪製曲線
    plot(x1, y1);
    legend('data','fit')
    title(' order: ',n)
    hold off;

end