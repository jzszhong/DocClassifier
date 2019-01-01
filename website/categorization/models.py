from django.db import models

class Document(models.Model):
    doc_id = models.CharField(max_length=10, primary_key=True)
    tags = models.CharField(max_length=30)
    
    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'document'

class Tag(models.Model):
    tag_id = models.CharField(max_length=10, primary_key=True)
    tag_name = models.CharField(max_length=128)
    
    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'tag'