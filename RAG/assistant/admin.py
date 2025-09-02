from django.contrib import admin
from .models import Document, DocumentChunk, ChatSession, ChatMessage, ProcessingTask


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'file_type', 'uploaded_by', 'uploaded_at', 'processed', 'total_chunks']
    list_filter = ['file_type', 'processed', 'uploaded_at']
    search_fields = ['title', 'uploaded_by__username']
    readonly_fields = ['id', 'uploaded_at', 'processed_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('uploaded_by')


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index', 'text_preview', 'created_at']
    list_filter = ['document__file_type', 'created_at']
    search_fields = ['document__title', 'text']
    readonly_fields = ['id', 'created_at', 'embedding_vector']
    
    def text_preview(self, obj):
        return obj.text[:100] + '...' if len(obj.text) > 100 else obj.text
    text_preview.short_description = 'Text Preview'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('document')


@admin.register(ChatSession) 
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'created_at', 'updated_at', 'message_count']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['title', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user').prefetch_related('messages')


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'message_type', 'content_preview', 'created_at']
    list_filter = ['message_type', 'created_at']
    search_fields = ['session__title', 'content']
    readonly_fields = ['id', 'created_at']
    
    def content_preview(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content Preview'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('session', 'session__user')


@admin.register(ProcessingTask)
class ProcessingTaskAdmin(admin.ModelAdmin):
    list_display = ['document', 'status', 'started_at', 'completed_at', 'celery_task_id']
    list_filter = ['status', 'started_at', 'completed_at']
    search_fields = ['document__title', 'celery_task_id']
    readonly_fields = ['id', 'started_at', 'completed_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('document')
