# Contributing

Please open an issue describing the proposed change and the behaviour it should improve.

Before submitting changes:

1. Run `python -m unittest discover -s tests -v`.
2. Run both example traces using the commands in the README.
3. Explain new assumptions and preserve the distinction between synthetic and hardware-measured outputs.
4. Add a focused test for changes to memory accounting or cache behaviour.

The runtime uses only the Python standard library. Discuss a new dependency before adding one. Do not include model weights, access tokens, private traces, or large raw logs in contributions.
