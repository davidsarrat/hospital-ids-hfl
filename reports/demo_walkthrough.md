# Demo Walkthrough

1. Install dependencies:

   ```bash
   make install
   ```

2. Download and clean CIC-IDS2017:

   ```bash
   make data
   ```

3. Create reproducible non-IID partitions:

   ```bash
   make partition SEED=123
   ```

4. Start the Flower deployment topology:

   ```bash
   make compose
   make flower-config
   make up
   ```

5. Run hierarchical training:

   ```bash
   make train GLOBAL_ROUNDS=3 REGIONAL_ROUNDS=2
   ```

6. Evaluate:

   ```bash
   make eval GLOBAL_ROUNDS=3
   ```

7. Check that `shared/` contains checkpoints and metrics only:

   ```bash
   find shared -type f
   ```
