# Shoe Price Prediction with CI/CD Pipeline

## Overview
This project implements a CI/CD pipeline for deploying a machine learning model that predicts shoe prices using the Datafiniti Women's Shoe dataset. The pipeline uses Jenkins, Docker, and GitHub workflows to automate code quality checks, unit testing, and deployment.

## Dataset and Preprocessing
The dataset contains information about women's shoes, including brand, categories, colors, and prices. The following preprocessing steps were performed:

- Compute the average price using minimum and maximum price columns
- Keep only relevant columns: brand, categories, colors, and price
- Handle missing values by dropping incomplete rows
- Encode categorical variables using label encoding
- Split the dataset into training (70 percent), validation (15 percent), and test (15 percent) sets
- Save the processed datasets in a dedicated directory

## Model Training
A random forest regressor was trained using the training dataset. The validation dataset was used to evaluate the model performance. The model was assessed using root mean squared error and r-squared score. Key steps include:

- Train a random forest model with 100 estimators
- Predict shoe prices on the validation set
- Calculate and print performance metrics
- Save the trained model for deployment
- Generate and save data visualizations, including price distribution, feature importance, and actual vs predicted prices

## CI/CD Pipeline
The project follows a structured CI/CD pipeline using Jenkins and GitHub workflows.

### Code Quality Enforcement
Flake8 is used to enforce Python coding standards. The workflow:

- Runs automatically on pushes to the development branch and pull requests from test to development
- Uses Flake8 to check for style violations
- Fails the workflow if issues are found to ensure only clean code is merged

### Unit Testing
A unit testing workflow using pytest ensures the correctness of the code. It:

- Triggers on pull requests to the test branch
- Runs automated tests for data loading, model loading, model performance, prediction range, and graph generation
- Ensures core functionalities work as expected

### Jenkins and Docker Integration
A Jenkinsfile defines the CI/CD pipeline, automating build, test, and deployment. The pipeline stages include:

1. Checkout Code: Pulls the latest repository code
2. Build Docker Image: Builds a containerized version of the model
3. Login to Docker Hub: Retrieves credentials and logs in
4. Push Image to Docker Hub: Uploads the model container
5. Cleanup: Removes the local Docker image to free space
6. Post-Build Notifications: Sends an email notification on success or failure

### Admin Notification Setup
Jenkins is configured to send email notifications about build status. This helps in quickly identifying deployment issues and keeping maintainers informed.

## Conclusion
This project implements a structured CI/CD pipeline for shoe price prediction. The combination of proper branching strategy, code quality enforcement, unit testing, Jenkins and Docker integration, and admin notifications ensures a reliable deployment process. This setup enhances code quality, reduces manual intervention, and accelerates the release cycle.

## Future Enhancements
- Implement automated security scans for Docker images
- Integrate performance monitoring after deployment
- Extend the pipeline for multi-environment deployments (development, staging, production)
