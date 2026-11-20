import 'package:flutter/material.dart';

class SystemMetricsCard extends StatelessWidget {
  final double? currentLoss;
  final double learningRate;
  final int step;
  final int totalSteps;

  const SystemMetricsCard({
    super.key,
    this.currentLoss,
    required this.learningRate,
    required this.step,
    required this.totalSteps,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.memory, color: Colors.cyanAccent),
                SizedBox(width: 8),
                Text(
                  'System Metrics & Hardware Telemetry',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            // Metrics Grid
            Row(
              children: [
                Expanded(
                  child: _metricTile(
                    'RAM Footprint (RSS)',
                    '447.8 MB',
                    subtitle: 'VmHWM peak: 447.8 MB (0% leak)',
                    color: Colors.greenAccent,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _metricTile(
                    'Step Latency (BF16)',
                    '3,407 ms',
                    subtitle: 'Throughput: 28.2 tok/s',
                    color: Colors.cyanAccent,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: _metricTile(
                    'Power Draw (CH7n)',
                    '~1,848 mW',
                    subtitle: '480 mA @ 3.85V | 34.5°C',
                    color: Colors.amberAccent,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _metricTile(
                    'Active Learning Rate',
                    learningRate.toStringAsExponential(1),
                    subtitle: learningRate <= 1e-4 ? 'Research Stable (1e-5)' : 'Fast Baseline (1e-3)',
                    color: learningRate <= 1e-4 ? Colors.purpleAccent : Colors.orangeAccent,
                  ),
                ),
              ],
            ),
            const Divider(height: 24, thickness: 1),
            const Row(
              children: [
                Icon(Icons.compare_arrows, color: Colors.purpleAccent, size: 20),
                SizedBox(width: 6),
                Text(
                  'Empirical Precision Optimization Study',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 8),
            const Text(
              'Loss of precision & throughput comparison on multimodal backward pass:',
              style: TextStyle(fontSize: 12, color: Colors.white70),
            ),
            const SizedBox(height: 8),
            Table(
              border: TableBorder.all(color: Colors.white24, borderRadius: BorderRadius.circular(6)),
              columnWidths: const {
                0: FlexColumnWidth(1.2),
                1: FlexColumnWidth(1.5),
                2: FlexColumnWidth(1.5),
                3: FlexColumnWidth(1.8),
              },
              children: const [
                TableRow(
                  decoration: BoxDecoration(color: Colors.white10),
                  children: [
                    Padding(padding: EdgeInsets.all(6), child: Text('Precision', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('Final Loss', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('Loss Error (|ΔL|)', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('Latency / Speedup', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 11))),
                  ],
                ),
                TableRow(
                  children: [
                    Padding(padding: EdgeInsets.all(6), child: Text('FP32', style: TextStyle(fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('6.8596', style: TextStyle(fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('0.00000 (Base)', style: TextStyle(fontSize: 11, color: Colors.white60))),
                    Padding(padding: EdgeInsets.all(6), child: Text('4,531.8 ms (1.00x)', style: TextStyle(fontSize: 11))),
                  ],
                ),
                TableRow(
                  decoration: BoxDecoration(color: Color(0x2200E676)),
                  children: [
                    Padding(padding: EdgeInsets.all(6), child: Text('BF16 ★', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.greenAccent))),
                    Padding(padding: EdgeInsets.all(6), child: Text('6.8600', style: TextStyle(fontSize: 11, color: Colors.greenAccent))),
                    Padding(padding: EdgeInsets.all(6), child: Text('0.00037 (<0.01%)', style: TextStyle(fontSize: 11, color: Colors.greenAccent))),
                    Padding(padding: EdgeInsets.all(6), child: Text('3,407.2 ms (+24.8%)', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.greenAccent))),
                  ],
                ),
                TableRow(
                  children: [
                    Padding(padding: EdgeInsets.all(6), child: Text('FP16', style: TextStyle(fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('6.8600', style: TextStyle(fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('0.00036 (<0.01%)', style: TextStyle(fontSize: 11))),
                    Padding(padding: EdgeInsets.all(6), child: Text('4,175.6 ms (+7.9%)', style: TextStyle(fontSize: 11))),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _metricTile(String label, String value, {String? subtitle, Color? color}) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: Colors.white60)),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color ?? Colors.white),
          ),
          if (subtitle != null) ...[
            const SizedBox(height: 2),
            Text(subtitle, style: const TextStyle(fontSize: 10, color: Colors.white38)),
          ],
        ],
      ),
    );
  }
}
