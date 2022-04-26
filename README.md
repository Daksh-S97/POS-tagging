# Assignment 2

**This assignment is designed to be completed in 3 weeks. Please start early!**

The entry point for this project is `pset2.ipynb`. It is a [Jupyter notebook](https://jupyter.org/) and we recommend installing [JupyterLab](https://jupyter.org/install.html) to run it. You can find a preview of the notebook by clicking the file on GitLab in a browser. To setup the computing resources and environments, see [this document](https://docs.google.com/document/d/1iuG6dNRAuhOU7K2ZeLNzIaV1w2InvMGq6VazBzZKFu4/edit?usp=sharing).

If you are new to pytorch, we recommend you to check out `optional_pytorch_basics.ipynb` first. This is fully optional and doesn't contain any deliverables. **You are required to read Eisenstein Ch. 7 before starting and during the assignment.**

**Due: May 13, 11:59 PM PST**

**Summary:** This project focuses on sequence labeling with Hidden Markov Models and Deep Learning models. The target domain is part-of-speech tagging on English and Norwegian from the Universal Dependencies dataset. You will:

- Do some basic preprocessing of the data
- Build a naive classifier that tags each word with its most common tag
- Implement a `Viterbi` Tagger using `Hidden Markov Model` in PyTorch
- Build a `Bi-LSTM` deep learning model using PyTorch
- Build a `Bi-LSTM_CRF` model using the above components (`Viterbi` and `Bi-LSTM`) 
- Implement techniques to improve your tagger

**Submission:** To submit this assignment, please refer to `submission_guide.md` for further details. Please don't forget to tag your submission. Failing to do so may result in a zero or some deductions on your project.

**Late Policy:** Each student will be granted 5 late days to use over the duration of the semester. You can use a maximum of 3 late days on any one project. **Weekends and holidays are also counted as late days.** Late submissions are automatically considered as using late days. Using late days will not affect your grade. However, projects submitted late after all late days have been used will receive no credit. Be careful!

**Academic honesty:** Homework assignments are to be completed individually and **you should not share your code**. E.g., it is not allowed to fork the public GitHub repo and push your progress there since others may see it. Verbal collaboration on homework assignments is acceptable, as well as re-implementation of relevant algorithms from research papers, but everything you turn in must be your own work, and you must note the names of anyone you collaborated with on each problem and cite resources that you used to learn about the problem. Suspected violations of academic integrity rules will be handled in accordance with the [UW guidelines on Student Conduct](https://www.washington.edu/cssc/for-students/overview-of-the-student-conduct-process/).
