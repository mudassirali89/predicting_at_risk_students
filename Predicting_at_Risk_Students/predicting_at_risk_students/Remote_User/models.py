from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class student_marks_model(models.Model):

    regno=models.CharField(max_length=300)
    names=models.CharField(max_length=300)
    sem1=models.CharField(max_length=300)
    sem2=models.CharField(max_length=300)
    sem3=models.CharField(max_length=300)


class student_risk_prediction_model(models.Model):

    regno = models.CharField(max_length=300)
    names = models.CharField(max_length=300)
    sem1 = models.CharField(max_length=300)
    sem2 = models.CharField(max_length=300)
    sem3 = models.CharField(max_length=300)
    avg= models.CharField(max_length=300)
    risk= models.CharField(max_length=300)

class detection_ratio_model(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


