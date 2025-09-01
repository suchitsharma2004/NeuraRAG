from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DocumentViewSet, ChatSessionViewSet, ProcessingTaskViewSet,
    ChatAPIView, search_documents, get_statistics, chat_interface,
    clear_vector_database, debug_documents, rebuild_vector_store
)

# Create router for ViewSets
router = DefaultRouter()
router.register(r'documents', DocumentViewSet)
router.register(r'chat-sessions', ChatSessionViewSet)
router.register(r'processing-tasks', ProcessingTaskViewSet)

# Define URL patterns
urlpatterns = [
    # Chat interface
    path('', chat_interface, name='chat-interface'),
    
    # API router URLs
    path('api/', include(router.urls)),
    
    # Additional API endpoints
    path('api/chat/', ChatAPIView.as_view(), name='chat-api'),
    path('api/search/', search_documents, name='search-documents'),
    path('api/stats/', get_statistics, name='statistics'),
    path('api/clear-vector-db/', clear_vector_database, name='clear-vector-database'),
    path('api/debug/', debug_documents, name='debug-documents'),
    path('api/rebuild-vector-store/', rebuild_vector_store, name='rebuild-vector-store'),
]
