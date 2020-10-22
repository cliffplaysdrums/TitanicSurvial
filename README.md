# Titanic: Machine Learning from Disaster
Below is an analysis of taking on Kaggle's [Titanic ML competition](https://www.kaggle.com/c/titanic/overview).
## Getting to know the data
The provided training data contains 892 samples and 9 features other than the target label. Thus, according to the curse of dimensionality, we may need to perform some transformation of our feature space, but it’s not so large that I’ll commit any effort to that yet. Instead, I'll look for correlation between features and labels and just get a feel for the data in general. It's also important to note that of our 892 training samples, 342 are survivors and 549 are not, so there is some imbalance in the data but not so significant that I'm overly concerned at this point.

Below is a plot showing passengers' sex and age and whether or not they survived. I've added random noise to the x-axis purely for visualization so that fewer points overlap. Clearly a higher percentage of women onboard survived than men, but there doesn't seem to be a strong correlation between age and survival.

![Sex vs Age plot](plots/Sex-Age.png)

Next is a plot showing passengers' sex again but this time with the class of their ticket. This time I've added noise to both axes as well as some transparency to the casualties since they dominate the space. We already know there is a correlation between sex and survival, but this plot indicates there may also be some correlation between ticket class and survival (look how few casualties there are in the female 1<sup>st</sup> & 2<sup>nd</sup> class ticket clusters).

![Sex vs Ticket Class plot](plots/Sex-Pclass.png)

Since there's visual correlation between survival and at least 2 attributes, let's go ahead and train a simple model and see how it performs. Since some of our features aren't easily converted to numbers (e.g. how does cabin 'C85' relate to cabin 'C23 C25 C27'?), and I want to start with a simple model, I'll select a subset of features to use for training. I initially chose ticket class, sex, age, count of siblings/spouse on board, count of parents/children on board, and ticket fare. However, it turns out we don't know the age of 177 passengers. Since data is somewhat limited for this exercise *and* we already saw that age doesn't have strong correlation, I'll omit it for now.