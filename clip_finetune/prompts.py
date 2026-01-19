# Galaxy10 DECaLS classification categories
s1 = 'disturbed galaxy'
s2 = 'merging galaxy'
s3 = 'round smooth galaxy'
s4 = 'in-between round smooth galaxy'
s5 = 'cigar-shaped smooth galaxy'
s6 = 'barred spiral galaxy'
s7 = 'unbarred tight spiral galaxy'
s8 = 'unbarred loose spiral galaxy'
s9 = 'edge-on galaxy without bulge'
s10 = 'edge-on galaxy with bulge'

# 保留完整的类别列表以保持代码兼容性
class_names = [globals()[f"s{i}"] for i in range(1, 11)]

# 为CLIP模型创建扩展的描述列表
class_descriptions_extended = [
    [
        "A telescope image of a galaxy with irregular, chaotic structure and distorted features.",
        "An astronomical view of a disturbed galaxy showing uneven arms and scattered star clusters.",
        "A deep-space image capturing a galaxy with warped, asymmetrical shapes.",
        "A cosmic snapshot of a star system with a messy, turbulent appearance."
    ],
    [
        "A telescope image of two galaxies colliding, with intertwined arms and bright cores.",
        "An astronomical scene of merging galaxies, showing tidal tails and overlapping structures.",
        "A deep-sky view of galaxies in the process of fusion, with glowing bridges.",
        "A celestial image of a galactic merger, blending starry cores and debris."
    ],
    [
        "A telescope image of a perfectly round, smooth galaxy with a soft, uniform glow.",
        "An astronomical view of a circular galaxy with no visible spiral arms, radiating evenly.",
        "A deep-space image of a smooth, spherical galaxy with a featureless core.",
        "A cosmic portrait of a round galaxy, glowing softly without distinct structures."
    ],
    [
        "A telescope image of a galaxy with a semi-round shape, blending smooth and faint spiral features.",
        "An astronomical view of a galaxy transitioning between round and spiral, with subtle arm-like patterns.",
        "A deep-sky image of a galaxy with a rounded core and hints of spiral structure.",
        "A celestial snapshot of a galaxy with a partially smooth, elliptical appearance."
    ],
    [
        "A telescope image of a long, narrow galaxy resembling a cigar, with a smooth, elongated glow.",
        "An astronomical view of a sleek, cigar-shaped galaxy with no visible spiral arms.",
        "A deep-space image of a slender, smooth galaxy stretched into an oval form.",
        "A cosmic view of a galaxy with a thin, streamlined shape and uniform brightness."
    ],
    [
        "A telescope image of a spiral galaxy with a bright central bar and prominent spiral arms.",
        "An astronomical scene of a galaxy featuring a distinct bar structure and tightly wound arms.",
        "A deep-sky view of a barred spiral galaxy with a glowing core and symmetrical arms.",
        "A celestial image of a spiral galaxy with a rectangular bar cutting through its arms."
    ],
    [
        "A telescope image of a spiral galaxy with tightly coiled arms and no central bar.",
        "An astronomical view of a galaxy with compact, well-defined spiral arms from a bright core.",
        "A deep-space image of an unbarred galaxy with intricate, tightly wound spiral patterns.",
        "A cosmic snapshot of a spiral galaxy with sharp, closely packed arms and a glowing center."
    ],
    [
        "A telescope image of a spiral galaxy with loosely wound arms and no central bar.",
        "An astronomical scene of a galaxy with open, flowing spiral arms extending from a bright core.",
        "A deep-sky view of an unbarred galaxy with relaxed, sweeping spiral arms.",
        "A celestial portrait of a spiral galaxy with wide, loosely coiled arms and a soft glow."
    ],
    [
        "A telescope image of a flat, edge-on galaxy with a thin disk and no central bulge.",
        "An astronomical view of a galaxy seen edge-on, appearing as a narrow band without a core.",
        "A deep-space image of a sleek, edge-on galaxy with a uniform disk and no bulge.",
        "A cosmic view of a galaxy aligned edge-on, showing a thin, featureless profile."
    ],
    [
        "A telescope image of an edge-on galaxy with a bright, bulging central core and a thin disk.",
        "An astronomical scene of a galaxy viewed edge-on, with a prominent bulge and extended arms.",
        "A deep-sky image of an edge-on galaxy with a glowing, swollen core and a flat disk.",
        "A celestial snapshot of a galaxy seen edge-on, with a distinct bulge at its center."
    ]
]

# 示例：选择每个类别的第一个描述用于 CLIP
class_descriptions0 = [descriptions[0] for descriptions in class_descriptions_extended]
class_descriptions1 = [descriptions[1] for descriptions in class_descriptions_extended]
class_descriptions2 = [descriptions[2] for descriptions in class_descriptions_extended]
class_descriptions3 = [descriptions[3] for descriptions in class_descriptions_extended]

# 或者，随机选择描述以增加多样性
import random
class_descriptions_random = [random.choice(descriptions) for descriptions in class_descriptions_extended]