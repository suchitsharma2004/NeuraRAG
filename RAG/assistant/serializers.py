from rest_framework import serializers
from .models import Document, DocumentChunk, ChatSession, ChatMessage, ProcessingTask


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for Document model"""
    uploaded_by = serializers.StringRelatedField(read_only=True)
    file_size = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = [
            'id', 'title', 'file', 'file_type', 'uploaded_by', 
            'uploaded_at', 'processed', 'processed_at', 'total_chunks', 'file_size'
        ]
        read_only_fields = ['id', 'uploaded_by', 'uploaded_at', 'processed', 'processed_at', 'total_chunks']
    
    def get_file_size(self, obj):
        """Get file size in human readable format"""
        if obj.file:
            size = obj.file.size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
        return "0 B"


class DocumentUploadSerializer(serializers.ModelSerializer):
    """Serializer for document upload"""
    class Meta:
        model = Document
        fields = ['title', 'file', 'file_type']
    
    def validate_file(self, value):
        """Validate uploaded file"""
        # Check file size (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 10MB")
        
        # Check file extension
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = value.name.lower().split('.')[-1]
        if f'.{file_extension}' not in allowed_extensions:
            raise serializers.ValidationError(
                f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        return value


class DocumentChunkSerializer(serializers.ModelSerializer):
    """Serializer for DocumentChunk model"""
    class Meta:
        model = DocumentChunk
        fields = ['id', 'chunk_index', 'text', 'created_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession model"""
    user = serializers.StringRelatedField(read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = ['id', 'user', 'title', 'created_at', 'updated_at', 'message_count']
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        """Get number of messages in the session"""
        return obj.messages.count()


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for ChatMessage model"""
    class Meta:
        model = ChatMessage
        fields = ['id', 'message_type', 'content', 'sources', 'created_at']
        read_only_fields = ['id', 'created_at']


class ChatQuerySerializer(serializers.Serializer):
    """Serializer for chat queries"""
    message = serializers.CharField(max_length=2000)
    session_id = serializers.UUIDField(required=False, allow_null=True)


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat responses"""
    session_id = serializers.UUIDField()
    user_message = ChatMessageSerializer()
    assistant_message = ChatMessageSerializer()


class ProcessingTaskSerializer(serializers.ModelSerializer):
    """Serializer for ProcessingTask model"""
    document = DocumentSerializer(read_only=True)
    
    class Meta:
        model = ProcessingTask
        fields = [
            'id', 'document', 'status', 'error_message', 
            'started_at', 'completed_at'
        ]
        read_only_fields = ['id', 'started_at', 'completed_at']
