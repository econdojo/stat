---
marp: true
math: mathjax
theme: default
size: 4:3
paginate: true
backgroundColor: '#f4f6fa'
backgroundImage: url('https://marp.app/assets/hero-background.svg')
header: 'Lecture 2: Prior Distributions'
footer: 'Fei Tan | Introduction to Bayesian Statistics'
style: |
  .logo {
    vertical-align: -0.2em;
  }
  section {
    color: #222;
    font-size: 24px;
    padding: 50px;
  }
  h1 {
    color: #003DA5;
    font-size: 38px;
    margin-bottom: 18px;
  }
  h2 {
    color: #003DA5;
    font-size: 30px;
    margin-bottom: 15px;
  }
  h3, h4, h5, h6 {
    color: #003DA5;
  }
  .slide-footer {
    color: #888;
  }
  .highlight {
    background-color: #ffeb3b;
    padding: 2px 4px;
    border-radius: 3px;
  }
  .code-box {
    background-color: #f5f5f5;
    border-radius: 10px;
    padding: 12px;
    margin: 12px 0;
    border: 1px solid #ddd;
    font-family: 'Courier New', monospace;
    font-size: 23px;
    line-height: 1.4;
  }
  table {
    margin: 15px auto;
    border-collapse: collapse;
    font-size: 19px;
  }
  table th, table td {
    border: 2px solid #003DA5;
    padding: 8px 12px;
    text-align: center;
  }
  table th {
    background-color: #003DA5;
    color: white;
  }
  ul, ol {
    margin: 10px 0;
    padding-left: 25px;
  }
  li {
    margin: 6px 0;
    line-height: 1.5;
  }
  p {
    margin: 10px 0;
    line-height: 1.5;
  }
  .equation-box {
    background-color: #f0f0f0;
    border-radius: 10px;
    padding: 15px;
    margin: 5px 0;
    border: 2px solid #003DA5;
    text-align: center;
    font-size: 20px;
  }
---

# Lecture 2: Prior Distributions

**Instructor:** Fei Tan

<img src="images/github.png" width="30" height="30" class="logo"> @econdojo &nbsp;&nbsp;&nbsp;&nbsp; <img src="images/youtube.png" width="30" height="30" class="logo"> @BusinessSchool101 &nbsp;&nbsp;&nbsp;&nbsp; <img src="images/slu.png" width="30" height="30" class="logo"> Saint Louis University

**Course:** Introduction to Bayesian Statistics  
**Date:** January 31, 2026

---

## The Road Ahead

1. [Preliminary](#normal-linear-regression)
2. [Specifying Prior Distributions](#conjugate-priors)

---

## Normal Linear Regression

**Model**

<div class="equation-box">

$$y_i=\beta_1x_{i1}+\cdots+\beta_Kx_{iK}+u_i,\quad u_i|x_i\sim_{i.i.d.}\mathcal{N}(0,\sigma^2),\quad i=1,\ldots,n$$

</div>

- Compact notation: $y=X\beta+u$, where

  $$y=
  \begin{bmatrix}
  y_{1} \\
  y_{2} \\
  \vdots \\
  y_{n}
  \end{bmatrix},\
  X=
  \begin{bmatrix}
  x_{11} & x_{12} & \cdots & x_{1K} \\
  x_{21} & x_{22} & \cdots & x_{2K} \\
  \vdots & \vdots & \ddots & \vdots \\
  x_{n1} & x_{n2} & \cdots & x_{nK}
  \end{bmatrix},\
  \beta=
  \begin{bmatrix}
  \beta_{1} \\
  \beta_{2} \\
  \vdots \\
  \beta_{K}
  \end{bmatrix},\
  u=
  \begin{bmatrix}
  u_{1} \\
  u_{2} \\
  \vdots \\
  u_{n}
  \end{bmatrix}$$

- Likelihood function

  $$f(y|\beta,\sigma^2)\propto\left(\frac{1}{\sigma^2}\right)^{n/2}\exp\left[-\frac{1}{2\sigma^2}(y-X\beta)'(y-X\beta)\right]$$

---

## Conjugate Priors

- Normal-inverse-gamma (type-2) prior

  $$\pi(\beta,\sigma^2)=\underbrace{\mathcal{N}(\beta|\beta_0,\sigma^2B_0)}_{\pi(\beta|\sigma^2)}\underbrace{\mathcal{IG}\text{-}2(\sigma^2|\alpha_0/2,\delta_0/2)}_{\pi(\sigma^2)}$$

- Exercise: posterior is of same family

  $$\pi(\beta,\sigma^2|y)=\underbrace{\mathcal{N}(\beta|\beta_1,\sigma^2B_1)}_{\pi(\beta|\sigma^2,y)}\underbrace{\mathcal{IG}\text{-}2(\sigma^2|\alpha_1/2,\delta_1/2)}_{\pi(\sigma^2|y)}$$

  where
  
  $$\begin{align*}
  B_1&=(X'X+B_0^{-1})^{-1} \\
  \beta_1 &= B_1(X'y+B_0^{-1}\beta_0) \\
  \alpha_1 &= \alpha_0+n \\
  \delta_1 &= \delta_0+y'y+\beta_0'B_0^{-1}\beta_0-\beta_1'B_1^{-1}\beta_1
  \end{align*}$$

- Exercise: $\pi(\beta|y)=t_{\alpha_1}(\beta_1,\frac{\delta_1}{\alpha_1}B_1)$, $\pi(\sigma^2|y)=\mathcal{IG}\text{-}2(\frac{\alpha_1}{2},\frac{\delta_1}{2})$

---

## (Im)proper Priors

- Proper priors integrate to unity, e.g.
  - $\pi(\sigma^2)=\mathcal{IG}\text{-}2(\alpha_0/2,\delta_0/2)$
  - equivalently, $h=1/\sigma^2$ (precision), $\pi(h)=\mathcal{G}(\alpha_0/2,\delta_0/2)$

- Improper priors are not integrable
  - improper vs. uninformative/diffuse prior
  - e.g. $\pi(\beta)\propto c>0$, $\pi(\sigma)\propto 1/\sigma$ (Jeffreys prior)
  - posterior may still be proper

- We work with proper prior
  - available information/methods to avoid improper prior
  - $m(y)$ based on improper prior can be manipulated

---

## Hierarchical Models

**Model**

<div class="equation-box">

$$\begin{align*}
\text{Hyperparameter prior:} && \alpha \sim \pi(\alpha|\delta) \\
\text{Parameter prior:} && \theta \sim \pi(\theta|\alpha) \\
\text{Likelihood:} && y \sim f(y|\theta)
\end{align*}$$

</div>

- Remarks
  - $f(y|\theta,\alpha)=f(y|\theta)$ $\Rightarrow$ $\alpha$ not identified
  - introduce $\alpha$ to facilitate computation/modeling

- Examples
  - Student-$t$ error: $u_i|x_i\sim_{i.i.d.}t_{\nu}(0,\sigma^2)$, $\nu\sim\pi(\nu|\nu_0)$
  - DSGE prior for VAR as will be covered later

---

## Training Sample Priors

**Bayesian updating**

<div class="equation-box">

$$\begin{align*}
\text{Training sample }y_1:&&\pi(\theta | y_1)\propto f(y_1 | \theta)\pi(\theta)\Rightarrow\underbrace{\color{blue}\pi(\theta|\alpha(y_1))}_{\text{posterior}} \\[5pt]
\text{Remaining sample }y_2:&&\pi(\theta | y_2,\alpha(y_1))\propto f(y_2 |\theta)\underbrace{\color{blue}\pi(\theta | \alpha(y_1))}_{\text{prior}}
\end{align*}$$

</div>

- Consider linear regression
  - improper prior: $\pi(\beta)\propto c$, $\pi(\sigma)\propto 1/\sigma$
  - proper joint posterior: $\beta|\sigma^2,y\sim\mathcal{N}(\hat{\beta},\sigma^2(X'X)^{-1})$, where $\hat{\beta}=(X'X)^{-1}X'y$, and $\sigma^2|y\sim\mathcal{IG}\text{-}2((n-K)/2,S^2/2)$, where $S^2=(y-X\hat{\beta})'(y-X\hat{\beta})$

---

## Conditionally Conjugate Priors

- Independent priors

  $$\pi(\beta,\sigma^2)=\underbrace{\mathcal{N}(\beta|\beta_0,B_0)}_{\pi(\beta)}\underbrace{\mathcal{IG}\text{-}2(\sigma^2|\alpha_0/2,\delta_0/2)}_{\pi(\sigma^2)}$$

- Exercise: *conditional* posteriors are of same family

  $$\pi(\beta|\sigma^2,y)=\mathcal{N}(\beta|\beta_1,B_1),\qquad\pi(\sigma^2|\beta,y)\propto\mathcal{IG}\text{-}2(\sigma^2|\alpha_1/2,\delta_1/2)$$

  where
  
  $$\begin{align*}
  B_1&=(\sigma^{-2}X'X+B_0^{-1})^{-1} \\
  \beta_1 &= B_1(\sigma^{-2}X'y+B_0^{-1}\beta_0) \\
  \alpha_1 &= \alpha_0+n \\
  \delta_1 &= \delta_0+(y-X\beta)'(y-X\beta)
  \end{align*}$$

---

## Readings

- Garthwaite, Kadane & O'Hagan (2005), "Statistical Methods for Eliciting Probability Distributions," *Journal of the American Statistical Association*

- O'Hagan et al. (2006), "Uncertain Judgements: Eliciting Experts' Probabilities," John Wiley & Sons
