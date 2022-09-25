clc; clear;

X = load('02HW1_Xtrain');
Y = load('02HW1_Ytrain');
f = figure;  
f.Position = [10 50 550 400]; 

for n=1:3
    subplot(1, 3, n)
    plot(X, Y, "o");
    hold on;
    
    p = polyfit(X ,Y, n);
    x1 = linspace(0, 1, 20);
    y1 = polyval(p, x1);
    
    plot(x1, y1);
    legend('data','fit')
    title(' order: ',n)
    hold off;

end

% plot(X,Y,'o',X,y1,'-')