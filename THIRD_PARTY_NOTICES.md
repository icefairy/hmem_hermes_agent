# Third-Party Notices

HMEM implements the following memory mechanisms inspired by the open-source
project listed below. HMEM adapts **design ideas and mechanisms**; where any
code, structure, or pattern is substantially derived from the original work,
the applicable MIT license is preserved below in accordance with its terms.

---

## dsh-memory · 灵枢（AEIS）× DeepSeek Harness

HMEM's memory mechanisms draw inspiration from:

- **Project**: [dsh-memory · 灵枢（AEIS）DeepSeek Harness 插件](https://github.com/FuRongJun-1999/dsh-memory)
- **Author**: FuRongJun-1999（荣）
- **Source**: https://github.com/FuRongJun-1999/dsh-memory
- **License**: MIT (reproduced below)
- **Inspired mechanisms**:
  - Auto-remember / auto-recall hooks (automatic memory sinking + pre-request injection)
  - Tiered five-layer memory (anchor / structure / knowledge / context / self)
  - Knowledge flywheel (verify → induce → associate → distill → derive)
  - Importance scoring + active (reversible) forgetting
  - Signal-to-noise dashboard (compression ratio / graph SNR)
  - Meta-cognitive (self-cognition) reflection

### MIT License

```text
MIT License

Copyright (c) 2026 FuRongJun-1999

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

> If any third-party code is later copied verbatim (or near-verbatim) into
> this repository, the corresponding copyright notice and license text must be
> preserved alongside that code per the MIT terms above.