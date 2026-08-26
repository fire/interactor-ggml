# interactor-forecast-litert

Side: `3-interactor` - placed at `3-interactor/forecast-litert` by the
[goal manifest](https://github.com/fire/fires-social-manager), and by nothing
else.

The LiteRT conversion spike, and whichever of its two outcomes it produces.

## Where this fits

```
repo init -u https://github.com/fire/fires-social-manager.git -b main
repo sync -j8
```

This repository is one of eleven. It is checked out beside its siblings, and
its `mix.exs` resolves them by path when the workspace exists and by
`github:` otherwise, so it builds alone in CI and hot-reloads in the
workspace.

## The data never leaves the machine

The store this project family builds holds real people's names, bios,
movements and social graph. Those people consented to none of it being
published. This repository is public and holds only code;
`gates/check_no_personal_data.py` in the manifest repository runs on
pre-commit and on pre-push, and blocks the push rather than trusting anyone to
remember.

## Licence

Apache-2.0 OR MIT, as two files, because dual licensing means the reader
chooses and one file cannot offer a choice.
