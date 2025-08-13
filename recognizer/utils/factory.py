"""
모듈 팩토리 패턴 구현

동적으로 모듈을 생성하고 관리하는 팩토리 클래스입니다.
새로운 모델을 쉽게 추가하고 교체할 수 있는 구조를 제공합니다.
"""

from typing import Dict, Any, Type, Optional
import importlib
import inspect
from abc import ABC
import logging

# 로깅 설정
logger = logging.getLogger(__name__)


class ModuleRegistry:
    """모듈 레지스트리 - 각 카테고리별로 모듈을 등록하고 관리"""
    
    def __init__(self, category_name: str):
        self.category_name = category_name
        self._modules: Dict[str, Type] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, module_class: Type, default_config: Optional[Dict[str, Any]] = None):
        """모듈 등록"""
        if not inspect.isclass(module_class):
            raise ValueError(f"Module must be a class, got {type(module_class)}")
        
        self._modules[name] = module_class
        if default_config:
            self._configs[name] = default_config
        
        logger.info(f"Registered {self.category_name} module: {name}")
    
    def create(self, name: str, config: Dict[str, Any], **kwargs) -> Any:
        """모듈 인스턴스 생성"""
        if name not in self._modules:
            available = list(self._modules.keys())
            raise ValueError(f"Unknown {self.category_name} module: {name}. Available: {available}")
        
        module_class = self._modules[name]
        
        # 기본 설정과 사용자 설정 병합
        final_config = self._configs.get(name, {}).copy()
        final_config.update(config)
        
        try:
            # 생성자 시그니처 확인
            sig = inspect.signature(module_class.__init__)
            
            # config를 첫 번째 인수로 받는지 확인
            if 'config' in sig.parameters:
                instance = module_class(config=final_config, **kwargs)
            else:
                # config의 각 키를 개별 인수로 전달
                filtered_kwargs = {}
                for param_name in sig.parameters.keys():
                    if param_name in final_config:
                        filtered_kwargs[param_name] = final_config[param_name]
                    elif param_name in kwargs:
                        filtered_kwargs[param_name] = kwargs[param_name]
                
                instance = module_class(**filtered_kwargs)
            
            logger.info(f"Created {self.category_name} module instance: {name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create {self.category_name} module {name}: {str(e)}")
            raise
    
    def list_modules(self) -> Dict[str, Type]:
        """등록된 모든 모듈 반환"""
        return self._modules.copy()
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """모듈의 기본 설정 반환"""
        return self._configs.get(name, {}).copy()
    
    def is_registered(self, name: str) -> bool:
        """모듈이 등록되어 있는지 확인"""
        return name in self._modules


class ModuleFactory:
    """통합 모듈 팩토리 - 모든 카테고리의 모듈을 관리"""
    
    _registries: Dict[str, ModuleRegistry] = {
        'pose_estimator': ModuleRegistry('pose_estimator'),
        'tracker': ModuleRegistry('tracker'),
        'scorer': ModuleRegistry('scorer'),
        'classifier': ModuleRegistry('classifier'),
    }
    
    @classmethod
    def get_registry(cls, category: str) -> ModuleRegistry:
        """카테고리별 레지스트리 반환"""
        if category not in cls._registries:
            raise ValueError(f"Unknown category: {category}. Available: {list(cls._registries.keys())}")
        return cls._registries[category]
    
    # Pose Estimator 관련 메서드
    @classmethod
    def register_pose_estimator(cls, name: str, estimator_class: Type, 
                              default_config: Optional[Dict[str, Any]] = None):
        """포즈 추정기 등록"""
        cls._registries['pose_estimator'].register(name, estimator_class, default_config)
    
    @classmethod
    def create_pose_estimator(cls, name: str, config: Dict[str, Any], **kwargs):
        """포즈 추정기 생성"""
        return cls._registries['pose_estimator'].create(name, config, **kwargs)
    
    @classmethod
    def list_pose_estimators(cls) -> Dict[str, Type]:
        """등록된 포즈 추정기 목록"""
        return cls._registries['pose_estimator'].list_modules()
    
    # Tracker 관련 메서드  
    @classmethod
    def register_tracker(cls, name: str, tracker_class: Type,
                        default_config: Optional[Dict[str, Any]] = None):
        """트래커 등록"""
        cls._registries['tracker'].register(name, tracker_class, default_config)
    
    @classmethod
    def create_tracker(cls, name: str, config: Dict[str, Any], **kwargs):
        """트래커 생성"""
        return cls._registries['tracker'].create(name, config, **kwargs)
    
    @classmethod
    def list_trackers(cls) -> Dict[str, Type]:
        """등록된 트래커 목록"""
        return cls._registries['tracker'].list_modules()
    
    # Scorer 관련 메서드
    @classmethod
    def register_scorer(cls, name: str, scorer_class: Type,
                       default_config: Optional[Dict[str, Any]] = None):
        """점수 계산기 등록"""
        cls._registries['scorer'].register(name, scorer_class, default_config)
    
    @classmethod
    def create_scorer(cls, name: str, config: Dict[str, Any], **kwargs):
        """점수 계산기 생성"""
        return cls._registries['scorer'].create(name, config, **kwargs)
    
    @classmethod
    def list_scorers(cls) -> Dict[str, Type]:
        """등록된 점수 계산기 목록"""
        return cls._registries['scorer'].list_modules()
    
    # Classifier 관련 메서드
    @classmethod
    def register_classifier(cls, name: str, classifier_class: Type,
                           default_config: Optional[Dict[str, Any]] = None):
        """분류기 등록"""
        cls._registries['classifier'].register(name, classifier_class, default_config)
    
    @classmethod
    def create_classifier(cls, name: str, config: Dict[str, Any], **kwargs):
        """분류기 생성"""
        return cls._registries['classifier'].create(name, config, **kwargs)
    
    @classmethod
    def list_classifiers(cls) -> Dict[str, Type]:
        """등록된 분류기 목록"""
        return cls._registries['classifier'].list_modules()
    
    # 통합 메서드들
    @classmethod
    def create_module(cls, category: str, name: str, config: Dict[str, Any], **kwargs):
        """일반적인 모듈 생성"""
        registry = cls.get_registry(category)
        return registry.create(name, config, **kwargs)
    
    @classmethod
    def register_module(cls, category: str, name: str, module_class: Type,
                       default_config: Optional[Dict[str, Any]] = None):
        """일반적인 모듈 등록"""
        registry = cls.get_registry(category)
        registry.register(name, module_class, default_config)
    
    @classmethod
    def list_all_modules(cls) -> Dict[str, Dict[str, Type]]:
        """모든 카테고리의 모든 모듈 목록"""
        return {
            category: registry.list_modules() 
            for category, registry in cls._registries.items()
        }
    
    @classmethod
    def get_module_info(cls, category: str, name: str) -> Dict[str, Any]:
        """모듈 정보 조회"""
        registry = cls.get_registry(category)
        if not registry.is_registered(name):
            return {}
        
        modules = registry.list_modules()
        config = registry.get_config(name)
        module_class = modules[name]
        
        return {
            'name': name,
            'category': category,
            'class': module_class,
            'module': module_class.__module__,
            'default_config': config,
            'docstring': module_class.__doc__,
            'methods': [method for method in dir(module_class) 
                       if not method.startswith('_') and callable(getattr(module_class, method))]
        }
    
    @classmethod
    def auto_discover_modules(cls, package_path: str):
        """자동으로 모듈 발견 및 등록"""
        try:
            # 패키지 경로에서 모듈들을 자동으로 찾아 등록
            # 이 기능은 추후 필요시 구현
            pass
        except ImportError as e:
            logger.warning(f"Failed to auto-discover modules in {package_path}: {str(e)}")
    
    @classmethod
    def validate_module(cls, category: str, name: str) -> bool:
        """모듈 유효성 검사"""
        try:
            registry = cls.get_registry(category)
            if not registry.is_registered(name):
                return False
            
            modules = registry.list_modules()
            module_class = modules[name]
            
            # 추상 메서드 검사 (ABC 상속 확인)
            if hasattr(module_class, '__abstractmethods__'):
                return len(module_class.__abstractmethods__) == 0
            
            return True
            
        except Exception as e:
            logger.error(f"Module validation failed for {category}.{name}: {str(e)}")
            return False
    
    @classmethod
    def create_from_config(cls, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인 설정에서 모든 모듈 생성"""
        modules = {}
        
        # 각 카테고리별로 모듈 생성
        category_mappings = {
            'pose_estimator': 'pose_config',
            'tracker': 'tracking_config', 
            'scorer': 'scoring_config',
            'classifier': 'classification_config'
        }
        
        for category, config_key in category_mappings.items():
            if config_key in pipeline_config:
                config = pipeline_config[config_key]
                if isinstance(config, dict) and 'name' in config:
                    module_name = config['name']
                    module_config = {k: v for k, v in config.items() if k != 'name'}
                    
                    try:
                        modules[category] = cls.create_module(category, module_name, module_config)
                        logger.info(f"Created {category}: {module_name}")
                    except Exception as e:
                        logger.error(f"Failed to create {category} {module_name}: {str(e)}")
                        raise
        
        return modules
    
    @classmethod
    def reset_registry(cls, category: Optional[str] = None):
        """레지스트리 초기화"""
        if category:
            if category in cls._registries:
                cls._registries[category] = ModuleRegistry(category)
        else:
            for cat in cls._registries.keys():
                cls._registries[cat] = ModuleRegistry(cat)
    
    @classmethod
    def get_factory_info(cls) -> Dict[str, Any]:
        """팩토리 전체 정보"""
        info = {
            'total_categories': len(cls._registries),
            'categories': {}
        }
        
        for category, registry in cls._registries.items():
            modules = registry.list_modules()
            info['categories'][category] = {
                'total_modules': len(modules),
                'modules': list(modules.keys())
            }
        
        return info


# 편의 함수들
def register_module_from_path(category: str, name: str, module_path: str, class_name: str,
                            default_config: Optional[Dict[str, Any]] = None):
    """모듈 경로에서 직접 클래스를 가져와 등록"""
    try:
        module = importlib.import_module(module_path)
        module_class = getattr(module, class_name)
        ModuleFactory.register_module(category, name, module_class, default_config)
        logger.info(f"Successfully registered {category}.{name} from {module_path}.{class_name}")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to register {category}.{name}: {str(e)}")
        raise


def create_module_with_fallback(category: str, primary_name: str, fallback_name: str,
                              config: Dict[str, Any], **kwargs):
    """기본 모듈로 생성하고, 실패하면 fallback 모듈 사용"""
    try:
        return ModuleFactory.create_module(category, primary_name, config, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to create {category}.{primary_name}, trying fallback: {fallback_name}")
        return ModuleFactory.create_module(category, fallback_name, config, **kwargs)


# 데코레이터
def register_module(category: str, name: str, default_config: Optional[Dict[str, Any]] = None):
    """클래스를 모듈로 등록하는 데코레이터"""
    def decorator(cls):
        ModuleFactory.register_module(category, name, cls, default_config)
        return cls
    return decorator