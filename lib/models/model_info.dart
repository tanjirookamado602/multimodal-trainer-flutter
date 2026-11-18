class ModelInfo {
  final String name;
  final String path;
  final String size;
  final bool isLoaded;

  ModelInfo({
    required this.name,
    required this.path,
    required this.size,
    required this.isLoaded,
  });
}

class ModelStatus {
  final bool isLoaded;
  final String modelName;
  final String size;
  final String memoryUsage;
  final String totalMemory;
  final String modelPath;

  ModelStatus({
    this.isLoaded = false,
    this.modelName = 'Qwen-Multimodal-Autograd (Tiny Backbone)',
    this.size = '64 hidden / 256 vocab',
    this.memoryUsage = '64 MB',
    this.totalMemory = '8 GB',
    this.modelPath = 'Native In-Memory C++ Model (hidden=64, layers=2)',
  });
}
