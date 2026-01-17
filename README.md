<h2>DQN 实战（CartPole 复现）</h2>
<p>原理参考：https://zhuanlan.zhihu.com/p/630554489<br>
策略网络：evaluate network 用用来计算策略选择的 Q 值和 Q 值迭代更新，梯度下降、反向传播的也是 evaluate network（在这应该是价值网络）<br>
目标网络：target network 用来计算 TD Target 中下一状态的 Q 值，网络参数更新来自 evaluate network 网络参数复制。<br>
设计 target network 目的是为了保持目标值稳定，防止过拟合，从而提高训练过程稳定和收敛速度<br>
梯度更新的是 evaluate network 的参数，不更新 target network，然后每隔一段时间将 evaluate network 的网络参数复制给 target network 网络参数，那么优化器 optimizer 设置的时候用的也是 evaluate network 的 parameters<br>
复制过程包括软更新和硬更新<p>
