## 回归问题

1. MSE（Mean squared loss):
   $$
   (y - f(x))^2
   $$

2. MAE Loss
  $$
  |y - f(x)|
  $$

3. Huber Loss

   Huber loss具备了MAE和MSE各自的优点，当δ趋向于0时它就退化成了MAE,而当δ趋向于无穷时则退化为了MSE
$$
\left\{
  \begin{aligned}
  &\frac{1}{2}[y - f(x)]^2 \\
  &\delta|y - f(x)| - \frac{1}{2} \delta^2
  
  \end{aligned}
  \right.
$$

