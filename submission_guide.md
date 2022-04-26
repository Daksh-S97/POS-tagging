# Submission Instructions
<a name="submission"></a>

We will be using `git`, a source code control tool, for distributing and submitting homework assignments in this class.
This will allow you to download the code and instruction for the homework, 
and also submit the labs in a standardized format that will streamline grading.

You will also be able to use `git` to commit your progress on the labs
as you go. This is **important**: Use `git` to back up your work. Back
up regularly by both committing and pushing your code as we describe below.

Course git repositories will be hosted as a repository in [CSE's
gitlab](https://gitlab.cs.washington.edu/), that is visible only to
you and the course staff.

## Getting started with `git`

There are numerous guides on using `git` that are available. They range from being interactive to just text-based. 
Find one that works and experiment -- making mistakes and fixing them is a great way to learn. 
Here is a [link to resources](https://help.github.com/articles/what-are-other-good-resources-for-learning-git-and-github) 
that GitHub suggests starting with.

Git may already be installed in your environment; if it's not, you'll need to install it first. 
For `bash`/Linux environments, git should be a simple `apt-get` / `yum` / etc. install. 
More detailed instructions may be [found here](http://git-scm.com/book/en/Getting-Started-Installing-Git).
Git is already installed on the CSE linux machines.

## Configuring `git` and cloning your repository for homework assignments

We will create a git repository for you to commit and submit your project. 
This repository is hosted on the [CSE's GitLab](https://gitlab.cs.washington.edu) , 
and you can view it by visiting the GitLab website at 
`https://gitlab.cs.washington.edu/cse447-sp2022/assignments/a1/CSE447-sp22-a[assignment number]-[your UW username]`. 

You'll have different repositories for each of the homework assignments this, 
so if you don't see this repository or are unable to access it, let us know immediately!

The first thing you'll need to do is set up a SSH key to allow communication with GitLab. Note that the general GitLab (`gitlab.com`) is different from the UW GitLab (`gitlab.cs.washington.edu`) we will be using, so you may need to still add your SSH key if you have not used UW GitLab before.

1.  If you don't already have one, generate a new SSH key. See [these instructions](http://doc.gitlab.com/ce/ssh/README.html) for details on how to do this. To check that your device is successfully connected to GitLab, run `ssh -T git@gitlab.cs.washington.edu` in your terminal.
2.  Visit the [GitLab SSH key management page](https://gitlab.cs.washington.edu/-/profile/keys). You'll need to log in using your CSE account.
3.  Click "Add SSH Key" and paste in your **public** key into the text area.

If you have already setup other SSH keys, you may need to manage your user and identity file using a `config` file under `.ssh` directory. For example, if you want to have different users and identify files for GitLab and Github, you can write a `config` file like below
```
Host github.com
    HostName github.com
    User [Your User Name]
    IdentityFile ~/.ssh/github_id_rsa

Host gitlab.cs.washington.edu
    HostName gitlab.cs.washington.edu
    User [Your User Name]
    IdentityFile ~/.ssh/gitlab_id_rsa
```

While you're logged into the GitLab website, browse around to see which projects you have access to. 
You should have access to `CSE447-sp22-a[assignment number]-[your UW username]`. 

We next want to move the code from the GitLab repository onto your local file system. 
To do this, you'll need to clone the repository by issuing the following commands on the command line:

```sh
$ cd [directory that you want to put your assignments]
$ git clone git@gitlab.cs.washington.edu:cse447-sp2022/assignments/a1/CSE447-sp22-a[assignment number]-[your UW username].git
$ cd CSE447-sp22-a[assignment number]-[your UW username]
```

This will make a complete replica of the repository locally. If you get an error that looks like:

```sh
Cloning into 'CSE447-sp22-a[assignment number]-[your UW username]'...
Permission denied (publickey).
fatal: Could not read from remote repository.
```

... then there is a problem with your GitLab configuration. Check to make sure that your GitLab username matches the repository suffix, that your private key is in your SSH directory (`~/.ssh`) and has the correct permissions, and that you can view the repository through the website.

Cloning will make a complete replica of the homework repository locally. Any time you `commit` and `push` your local changes, they will appear in the GitLab repository.  Since we'll be grading the copy in the GitLab repository, it's important that you remember to push all of your changes!

## Submitting your assignment

You may submit your code multiple times; we will use the latest version you submit that arrives 
before the deadline. 

**Important**: In order for your write-up to be added to the git repo, you need to explicitly add it:

```sh
$ git add writeup.pdf ...
```

Or if you do
```sh
$ git add .
```

Then it will add *all* the files inside your directory to the repo. This is generally a bad practice if you have other unrelevant files or hidden files in the directories. Be sure to add the files that you do not want to submit into `.gitignore`, such that the file will not be included in your final submission.
Finally, you need to commit and push the changes:

```sh
$ git commit -a -m 'my latest changes are here (or any message you want)'
$ git push
```

The flag `-a` means "commit all changes" (the easiest way); you can also manually select which files you want to commit. You can run `git status` to check the status of your changes.  Commit and push as often as you want to save your homework on gitlab, before the deadline.  **When you want to submit, you should also add the tag `a[assignment]-final` before pushing**. For example, when you are submitting project 1, you should run

```sh
$ git tag a1-final
```

The criteria for your assignment being submitted on time is that your code must be tagged and pushed by the due date and time. This means that if one of the TAs or the instructor were to open up GitLab, they would be able to see your solutions on the GitLab web page.

**Just because your code has been committed on your local machine does not mean that it has been submitted -- it needs to be on GitLab!**

The tag can only be created once; if you want to update your submission, you need to delete it, then re-tag.  There is convenient a bash script `submit.sh` in the root level directory of your repository that does this automatically.  You first commit, then run `submit.sh`, which  deletes any prior tag for the current lab, tags the current commit, and pushes the branch and tag to GitLab. If you are using Linux or Mac OSX, you should be able to run the following (note that the assignment number is something like 1 instead of a1):

```sh
$ bash submit.sh [assignment number]
```

**Alternatively**, you can also push your changes to any branch, then create a tag using `GitLab` interface. To do this, you should click in the following order on the page of your repository: `Repository` -> `Tags` -> `New Tag`. Input your `tag name` and the branch you would like to create from. Then click `Create Tag` to finish your submission.

### Final Word of Caution!

Git is a distributed version control system. This means everything operates offline until you run `git pull` or `git push`. This is a great feature.

The bad thing is that you may **forget to `git push` your changes**. This is why we strongly, strongly suggest that you **check GitLab to be sure that what you want us to see matches up with what you expect**.