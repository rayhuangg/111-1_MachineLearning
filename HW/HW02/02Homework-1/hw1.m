clc; clear;

X = load('02HW1_Xtrain');
Y = load('02HW1_Ytrain');
plot(X,Y, "o");
hold on;

p = polyfit(X ,Y, 2);
y1 = polyval(p, X);

plot(X, y1, '-')
hold off;


% plot(X,Y,'o',X,y1,'-')