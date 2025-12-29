---
marp: true
math: mathjax
theme: default
size: 4:3
paginate: true
backgroundColor: '#f4f6fa'
backgroundImage: url('https://marp.app/assets/hero-background.svg')
header: 'Lecture 0: Basic Concepts of Probability and Inference'
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
    margin: 12px 0;
    border: 2px solid #003DA5;
    text-align: center;
    font-size: 26px;
  }
---

# Lecture 0: Basic Concepts of Probability and Inference

**Instructor:** Fei Tan

<img src="images/github.png" width="30" height="30" class="logo"> @econdojo &nbsp;&nbsp;&nbsp;&nbsp; <img src="images/youtube.png" width="30" height="30" class="logo"> @BusinessSchool101 &nbsp;&nbsp;&nbsp;&nbsp; <img src="images/slu.png" width="30" height="30" class="logo"> Saint Louis University

**Course:** Introduction to Bayesian Statistics  
**Date:** December 28, 2025

---

## What Is the Course About?

- Introduce Bayesian inferential methods & develop hands-on skills for Python data science

- **Why Bayesian paradigm?** Handle sophisticated models & uncertainty in decision making

- Main references
  - required: Greenberg (2008), "*Introduction to Bayesian Econometrics*"
  - optional: Geweke (2005), "*Contemporary Bayesian Econometrics and Statistics*"

- Homework production via [Visual Studio Code](https://code.visualstudio.com/)
  - LaTeX typesetting
  - Python programming

---

## The Road Ahead

1. [Probability](#frequentist-vs-bayesian)
2. [Prior, Likelihood, and Posterior](#prior-likelihood-and-posterior)

---

## Frequentist v.s. Bayesian

**Probability axioms**

<div class="equation-box">

$$\begin{align*}
&1.\ 0\leq\mathbb{P}(A)\leq 1\quad\text{for any event }A\\
&2.\ \mathbb{P}(A)=1\quad\text{if event }A\text{ represents logical truth}\\
&3.\ \mathbb{P}(A\cup B)=\mathbb{P}(A)+\mathbb{P}(B)\quad\text{for disjoint events }A\text{ and }B\\
&4.\ \mathbb{P}(A|B)=\mathbb{P}(A\cap B)/\mathbb{P}(B)\quad\text{(conditional probability)}
\end{align*}$$

</div>

- Satisfied by any assignment of probabilities
  - frequentists assign probabilities to events describing outcome of *repeated* experiment
  - Bayesians assign 'subjective' probabilities to uncertain events [de Finetti's (1990) coherency principle]

- How likely it rains tomorrow?

---

## Prior, Likelihood, and Posterior

**Bayes theorem**

<div class="equation-box">

$$\pi(\theta|y)=\frac{f(y|\theta)\pi(\theta)}{m(y)}\propto f(y|\theta)\pi(\theta)$$

</div>

- Bayesians treat **parameters** $\theta$ as random variables & **data** $y=[y_1,\ldots,y_n]'$ as given
  - start with **prior** density $\pi(\theta)$
  - update by **likelihood** function $f(y|\theta)$
  - **posterior** density $\pi(\theta|y)$ proportional to prior $\times$ likelihood
  - **marginal likelihood** $m(y)=\int f(y|\theta)\pi(\theta)d\theta$

---

## Coin-Tossing Example

- Likelihood function

    - one toss (Bernoulli): $\mathbb{P}(Y_i=1)=\theta=1-\mathbb{P}(Y_i=0)$

    $$f(y_i|\theta)=\theta^{y_i}(1-\theta)^{1-y_i}$$

    - $n$ independent tosses

    $$f(y_1,\ldots,y_n|\theta)=\theta^{\sum y_i}(1-\theta)^{n-\sum y_i}$$

- (Conjugate) beta prior: $\theta\sim\mathcal{B}(\alpha,\beta)$

    $$\pi(\theta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1},\quad 0\leq\theta\leq1,\quad\alpha,\beta>0$$

- Beta posterior: $\theta | y\sim\mathcal{B}(\alpha+\sum y_i,\beta+n-\sum y_i)$

    $$\pi(\theta|y)\propto\theta^{\alpha+\sum y_i-1}(1-\theta)^{\beta+n-\sum y_i-1}$$

---

## Hyperparameters

<img src="images/lec0/Fig1.png" width="70%" style="display: block; margin: 0 auto;">

- Shape of beta: $\mathbb{E}(\theta)=\frac{\alpha}{\alpha+\beta},\quad \mathbb{V}(\theta)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

---

## Sample Size

<img src="images/lec0/Fig2.png" width="70%" style="display: block; margin: 0 auto;">

- $\mathbb{E}(\theta|y)=\frac{\alpha+\beta}{\alpha+\beta+n}\mathbb{E}(\theta)+\frac{n}{\alpha+\beta+n}\bar{y}\to_{n\to\infty}\bar{y}\text{ (MLE)}$

---

## References

- de Finetti (1990), "Theory of Probability", John Wiley & Sons
