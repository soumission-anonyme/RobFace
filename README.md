# RobFace

Face recognition is a common authentication technology in practice, which requires high adversarial robustness. It is thus desirable to have an efficient and easy-to-use method for evaluating the robustness of (possibly third-party) trained face recognition systems. Existing approaches to evaluating robustness of face recognition systems are either based on empirical evaluation (e.g., measuring attacking success rate using state-of-the-art attacking methods) or formal analysis (e.g., measuring the Lipschitz constant). While the former demands significant user-efforts and expertise, the latter is extremely time-consuming. In pursuit of a comprehensive, efficient, easy-to-use and scalable estimation of the robustness of face recognition systems, we take an old-school alternative approach and introduce RobFace, i.e., an optimised test suite containing transferable adversarial face images which are designed to comprehensively evaluate a face recognition system's robustness along a variety of dimensions. RobFace is model-agnostic and yet tuned such that it provides robustness estimation which is consistent with model-specific empirical evaluation or formal analysis. We support this claim through extensive experimental results with various perturbations on multiple face recognition systems. To our knowledge, RobFace is the first model-agnostic robustness estimation test suite.

About the name:
* "ROBust" - This approach is used to evaluate robustness
* "FACE" - This is for face recognition systems

# What's New
- [Sep. 2022] [ICSE anonymous submission]([https://arxiv.org/pdf/2103.14030.pdf](https://icse2023.hotcrp.com/)) is based on the code and sample data provided in this repository

# Requirements (for test suite construction)
* python >= 3.7.1
* pytorch >= 1.1.0
* torchvision >= 0.3.0 

# Requirements (for using the test suite)
Just your computer and your own model will suffice!
The test suite is strongly applicable to various systems.

# Demonstration Test Suite Construction Examples
* Demo part 1: to generate a large number of samples
* Demo part 2: to optimise the best composition of test suite
* Demo part 3 & 4: extra experimental details.


## Face recognition systems
See [README.md](face_sdk/README.md) in [face_sdk](face_sdk), which comes from external repositories.
