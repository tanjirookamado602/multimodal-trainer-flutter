import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/model_info.dart';
import '../providers/model_provider.dart';
import '../providers/validation_provider.dart';

class ModelStatusCard extends StatelessWidget {
  final ModelStatus status;

  const ModelStatusCard({super.key, required this.status});

  void _showPathDialog(BuildContext context, ModelProvider modelProvider) {
    final controller = TextEditingController(text: status.modelPath);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Model Architecture Details'),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Backbone: Qwen Architecture (Tiny In-Memory)\n'
              '• Hidden Size: 64\n'
              '• Intermediate Size: 128\n'
              '• Layers: 2\n'
              '• Attention Heads: 4 (KV heads: 2)\n'
              '• Vocab Size: 256 tokens\n'
              '• Trainable Projector: W_proj [64, 32]\n\n'
              'Note: Base transformer weights are frozen. Only the multimodal visual projection matrix is updated via backward pass.',
              style: TextStyle(fontSize: 13, color: Colors.white70),
            ),
          ],
        ),
        actions: [
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final modelProvider = Provider.of<ModelProvider>(context, listen: false);

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Row(
                    children: [
                      Icon(
                        status.isLoaded ? Icons.check_circle : Icons.model_training,
                        color: status.isLoaded ? Colors.greenAccent : Colors.orangeAccent,
                      ),
                      const SizedBox(width: 8),
                      Flexible(
                        child: Text(
                          'Backbone: ${status.modelName}',
                          style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.info_outline),
                  tooltip: 'Model Info',
                  onPressed: () => _showPathDialog(context, modelProvider),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text('Architecture: ${status.size} (2 layers, 4 attention heads)'),
            Text('Device Memory: ${status.memoryUsage} active / ${status.totalMemory} total'),
            Text('Model Source: ${status.modelPath}', overflow: TextOverflow.ellipsis),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                ElevatedButton.icon(
                  onPressed: () => modelProvider.loadModel(status.modelPath),
                  icon: const Icon(Icons.refresh, size: 18),
                  label: const Text('Initialize Model Context'),
                ),
                OutlinedButton.icon(
                  onPressed: () {
                    final validationProvider = Provider.of<ValidationProvider>(context, listen: false);
                    validationProvider.runValidation(status.modelPath, isPreTraining: true);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Running SATE AI Stress Test... check SATE AI tab for report'),
                        backgroundColor: Colors.deepPurple,
                        duration: Duration(seconds: 2),
                      ),
                    );
                  },
                  icon: const Icon(Icons.shield, size: 18, color: Colors.purpleAccent),
                  label: const Text('SATE AI Stress Test'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
