# Interpretable Machine Learning 2020

Lecture notes for 'Interpretable Machine Learning' at [WUT](https://usosweb.usos.pw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&kod=1120-IN000-MSP-0501) and [UoW](https://usosweb.uw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&kod=1000-1M18WUM). Summer semester 2019/2020

This document: http://tiny.cc/IML2020

Slack for this course: http://iml2020workspace.slack.com

# Introduction

The course consists of a lecture, computer laboratory and project.

The course is elective. The rules of passing may seem non-standard. Make sure that you understand them to avoid unpleasant consequences.
I believe that one of the most important skills in building ML/XAI models is flexibility and a proactive approach to the problem. In this course, the assessment criteria will strongly reward both flexibility and a proactive approach.

# Design Principles

The design of this course is based on four principles:

- Mixing experiences during studies is good. It allows you to generate more ideas. Also, in mixed groups, we can improve our communication skills,
- In XAI, the interface/esthetic of the solution is important. XAI, like earlier HCI (Human Computer Interaction), is on the borderline between technical, domain and cognitive aspects. Therefore, apart from the purely technical descriptions, the results must be grounded in the domain and are communicated aesthetically and legibly, 
- communication of results is important. Both in science and business, it is very important to be able to present the results concisely and legibly. In this course, it should translate into the ability to describe one XAI story in the form of a short chapter/article.
- It is worth doing useful things. Let's look for new applications for XAI methods discussed on typical predictive problems.


# Meetings

Plan for the summer semester 2019/2020. WUT classes are on Thursdays, UoW classes are on Fridays. We will meet online here: meet.google.com/yfq-hckf-pgu.


* 2020-02-27/28 -- Introduction
* 2020-03-05/06 -- Break Down / SHAP. [EMA chapter](https://pbiecek.github.io/ema/breakDown.html), [paper shap](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions), [paper break down](https://arxiv.org/abs/1903.11420)
* 2020-03-12/13 -- [XAI stories: first meeting, groups are assembles]
* 2020-03-19/20 -- LIME. [EMA chapter](https://pbiecek.github.io/ema/LIME.html), [paper lime](https://arxiv.org/abs/1602.04938)
* 2020-03-26/27 -- Ceteris Paribus profiles / Partial Dependence profiles. [EMA chapter](https://pbiecek.github.io/ema/partialDependenceProfiles.html), [paper pdp/ale](https://cran.r-project.org/web/packages/ALEPlot/vignettes/AccumulatedLocalEffectPlot.pdf)
* 2020-04-02/03 -- [XAI stories: first version of the solution]
* 2020-04-09 --  TBA
* 2020-04-16/17 -- Variable's importance. [EMA chapter](https://pbiecek.github.io/ema/featureImportance.html), [paper pvi](https://arxiv.org/abs/1801.01489)
* 2020-04-23/24 -- [XAI stories: second version of the solution]
* 2020-04-30 --  TBA
* 2020-05-07 --  TBA
* 2020-05-14/15 -- Model diagnostic plots. [EMA chapter](https://pbiecek.github.io/ema/residualDiagnostic.html), [paper auditor](https://arxiv.org/abs/1809.07763)
* 2020-05-21/22 -- students presentations
* 2020-05-28/29 -- students presentations
* 2020-06-04/05 -- [XAI stories: final version of the solution]

# How to get a good grade

From different activities, you can get from 0 to 100 points. 51 points are needed to pass this course.
There are three key components.

Chapter in the 'XAI stories' [0-60 points]
 - quality of trained predictive models [0-10 points]
 - quality of dataset level explanations [0-10 points]
 - quality of instance level explanations [0-10 points]
 - quality of the charts/visuals/diagrams [0-10 points]
 - the relevance of the example [0-10 points]
 - presentation of key results during the final meeting [0-10 points]

Presentation of a selected XAI related article [0-10 points]

Home works [0-30 points]
 - home work 1 for 0-5 points: Train a predictive model for selected ML problem (see issues). Submit knitr/notebook script to GitHub (directory Homeworks/H1/FirstnameLastname). **Deadline: 2020-03-12**
 - [home work 2](https://github.com/pbiecek/InterpretableMachineLearning2020/blob/master/Homeworks/H2/README.md) for 0-5 points. **Deadline: 2020-03-26**
 - [home work 3](https://github.com/pbiecek/InterpretableMachineLearning2020/blob/master/Homeworks/H3/README.md) for 0-5 points. **Deadline: 2020-04-02**
 - [home work 4](https://github.com/pbiecek/InterpretableMachineLearning2020/blob/master/Homeworks/H4/README.md) for 0-5 points. **Deadline: 2020-04-16**
 - [home work 5](https://github.com/pbiecek/InterpretableMachineLearning2020/blob/master/Homeworks/H5/README.md) for 0-5 points. **Deadline: 2020-04-30**
 - [home work 6](https://github.com/pbiecek/InterpretableMachineLearning2020/blob/master/Homeworks/H6/README.md) for 0-5 points. **Deadline: 2020-05-14**


# Presentations

Presentations can be prepared by one or two students. Each group should present a single XAI related paper (journal or conference). Each group should choose a different paper. Here are some suggestions.

* [Revealing the Dark Secrets of BERT](https://arxiv.org/abs/1908.08593) note, that you need to have some knowledge related to NLP to take this topic,
* [Explanation in Artificial Intelligence: Insights from the Social Sciences](https://arxiv.org/pdf/1706.07269.pdf) by Tim Miller
* [Explainable AI for Trees: From Local Explanations to Global Understanding](https://arxiv.org/abs/1905.04610) by Scott Lundberg
* [One Explanation Does Not Fit All: A Toolkit and Taxonomy of AI Explainability Techniques](https://arxiv.org/abs/1909.03012) by Vijay Arya et al
* [iNNvestigate neural networks](https://arxiv.org/abs/1808.04260) by Maximilian Alber et al
* [Veridical Data Science](https://slideslive.com/38921720/veridical-data-science) by Bin Yu at NIPS19
* [Agency + Automation: Designing Artificial Intelligence into Interactive Systems
Jeff Heer](https://slideslive.com/38921798/agency-automation-designing-artificial-intelligence-into-interactive-systems) by Jeff Heer at NIPS19
* other papers from this list https://github.com/pbiecek/xai_resources



# Projects

Project proposals are described as issues in this repository. Each issue is a single problem in which you need to train a few predictive models and explain them. Among different issues, you will fond applications in different areas, some concern medical data, some concern financial data. 

Each group of students should choose one issue they want to solve. After consultation with the lecturer, you can also submit your projects.
Projects should be solved in groups. The ideal group consists of three people, one student from each university (PW, US, SGH).
Data Scientists from McKinsey will help us with these projects. More details about the rules of cooperation will be given during classes.

The project ends with a small article prepared in English and a short presentation summarizing the key results. The study will be available to the public in the form of open-gitbook. 

See [Limitations of Interpretable Machine Learning Methods](https://compstat-lmu.github.io/iml_methods_limitations/) as an example to follow. During this course, we are going to gather several use-cases/success stories for explainable machine learning.

## Phase 1

After the first meeting, each group should:

- know what problem they'll be working on,
- know how to communicate with every team member (own slack channel, something else?)
- initially share/distribute work on (1) finding similar solutions in literature, (2) generating models, (3) generating explanations, (4) describing models and explanations, 
- establish an internal work schedule for the next meeting (already in 3 weeks).


# Literature

The literature will be added on an ongoing basis. 

* [Explanatory Model Analysis. Explore, Explain and Examine Predictive Models](https://pbiecek.github.io/ema/)
* [Interpretable Machine Learning. A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/)
