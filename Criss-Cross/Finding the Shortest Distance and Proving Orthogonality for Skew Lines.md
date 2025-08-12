# Finding the Shortest Distance and Proving Orthogonality for Skew Lines

The core of this problem is to find the minimum distance between two lines that do not intersect and are not parallel (skew lines). The method involves using a difference vector and applying a fundamental principle of calculus to find its minimum magnitude. This minimum magnitude corresponds to the shortest distance.

### Step 1: Defining the Difference Vector

The first step is to define the difference vector, $d(t, s)$, between an arbitrary point on Line 1 , $x_1(t)$, and an arbitrary point on Line $2, x_2(s)$.

Given the lines:

$$
\begin{aligned}
& x_1(t)=t e_1+(2-t) e_2 \\
& x_2(s)=2 s e_1-e_2-s e_3
\end{aligned}
$$


The difference vector is:

$$
d(t, s)=x_1(t)-x_2(s)
$$


Substituting the expressions for the lines, we get:

$$
d(t, s)=\left(t e_1+(2-t) e_2\right)-\left(2 s e_1-e_2-s e_3\right)
$$

$$
d(t, s)=(t-2 s) e_1+(2-t-(-1)) e_2+(0-(-s)) e_3
$$


$$
d(t, s)=(t-2 s) e_1+(3-t) e_2+s e_3
$$

### Step 2: Finding the Shortest Distance

The shortest distance is the minimum magnitude of the difference vector, $|d|$. To find this minimum, we can minimize the squared magnitude, $|d|^2$, which simplifies the math.
$$
|d|^2=(t-2 s)^2+(3-t)^2+s^2
$$


To find the values of $t$ and $s$ that minimize this function, we take the partial derivatives with respect to $t$ and $s$ and set them to zero:

Partial derivative with respect to $t$ :

$$
\begin{gathered}
\frac{\partial|d|^2}{\partial t}=2(t-2 s)(1)+2(3-t)(-1)=0 \\
t-2 s-3+t=0 \Longrightarrow 2 t-2 s=3
\end{gathered}
$$


Partial derivative with respect to $s$ :

$$
\begin{gathered}
\frac{\partial|d|^2}{\partial s}=2(t-2 s)(-2)+2 s(1)=0 \\
-4 t+8 s+2 s=0 \Longrightarrow-4 t+10 s=0 \Longrightarrow 2 t=5 s
\end{gathered}
$$


Now we have a system of two linear equations:
1. $2 t-2 s=3$
2. $2 t=5 s$

Substitute equation (2) into equation (1):
$$
5 s-2 s=3 \Longrightarrow 3 s=3 \Longrightarrow s=1
$$


Substitute $s=1$ back into equation (2):

$$
2 t=5(1) \Longrightarrow t=\frac{5}{2}=2.5
$$


The shortest distance occurs at $t=2.5$ and $s=1$.
Now, substitute these values back into the difference vector to find the shortest distance vector and its magnitude:

$$
\begin{gathered}
d(2.5,1)=(2.5-2(1)) e_1+(3-2.5) e_2+(1) e_3 \\
d(2.5,1)=0.5 e_1+0.5 e_2+1 e_3
\end{gathered}
$$


The shortest distance is the magnitude of this vector:

$$
|d|=\sqrt{(0.5)^2+(0.5)^2+(1)^2}=\sqrt{0.25+0.25+1}=\sqrt{1.5} \approx 1.2247
$$

### Step 3: Proving Orthogonality

A crucial property of the shortest distance between two skew lines is that the difference vector at this minimum is orthogonal to the tangent vectors of both lines.

First, let's find the tangent vectors by differentiating the line equations:

$$
\begin{aligned}
& t_1=\frac{d x_1}{d t}=e_1-e_2 \\
& t_2=\frac{d x_2}{d s}=2 e_1-e_3
\end{aligned}
$$


Note that these vectors are constant, so they do not depend on $t$ or $s$.
Now, we check if the difference vector at the shortest distance, $d(2.5,1)$, is orthogonal to both tangent vectors by checking if their dot products are zero.

Orthogonality with $t_1$ :

$$
\begin{aligned}
& d(2.5,1) \cdot t_1=\left(0.5 e_1+0.5 e_2+1 e_3\right) \cdot\left(e_1-e_2\right) \\
= & (0.5)(1)+(0.5)(-1)+(1)(0)=0.5-0.5+0=0
\end{aligned}
$$


Orthogonality with $t_2$ :

$$
d(2.5,1) \cdot t_2=\left(0.5 e_1+0.5 e_2+1 e_3\right) \cdot\left(2 e_1-e_3\right)
$$

$$
=(0.5)(2)+(0.5)(0)+(1)(-1)=1+0-1=0
$$


Since both dot products are zero, the difference vector $d(2.5,1)$ is indeed orthogonal to the tangent vectors of both lines, proving the condition.
