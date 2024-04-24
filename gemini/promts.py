instruccion = """1"""

instruccion2 = """Eres un asistente virtual de la tienda Rojo Bermelo.
            ------------------
            Directrices de la información:
            - Es muy importante que no inventes artículos, si no hay coincidencias, no inventes artículos.
            - Siempre comienza la interacción con un saludo.
            - Vendemos : Art Toys, Accesorios y Cocina.
            - Contesta de manera casual.
            - Responde con emojis.
            - insentiva al usuario a comprar.
            --------------------
            Responde **unicamente** y **solamente** con la siguiente información:
            """


DOCUMENT1 = {
    "title": "Art Toys",
    "content": """-  andy warhol dunny series
    • Colecciónalos todos!Andy Warhol «The Mark of the Beast» Dunny – 1/48 (rare)Andy Warhol «Mao» Dunny – 1/24Andy Warhol «Camouflage» Dunny – 1/24Andy Warhol «Campbell’s Soup Can» Dunny – 3/24Andy Warhol «Cow» Dunny – 1/24Andy Warhol «Dollar Sign» Dunny – 3/24Andy Warhol «$1.57 Giant Size» Dunny – 2/24Andy Warhol «Flowers» Dunny – 2/24Andy Warhol «Elvis» Dunny – 1/48 (rare)Andy Warhol «Banana» Dunny – 3/24Andy Warhol «Fragile Handle With Care» Dunny – 1/24Andy Warhol «Skulls» Dunny – 2/24Andy Warhol «Campbell’s Tomato Juice Box» Dunny – 3/24Andy Warhol «Gun» Dunny – 1/24
-  azure superjanky x junk mizuno
    • La legendaria artista del manga Junko Mizuno nos trae esta pieza llamada Azure Ailurophile Superjanky de 8 pulgadas
-  calorie sushi seaweed
    • Artista: JiaMaterial: PlásticoMedida: 10 cm
-  captain cornstarch x ron english
    • De la serie de Ron English «Cereal Killers» traemos Captain Cornstarch, de 9 pulgadas.
-  count calorie monotone x ron english
    • De la serie de Ron English «Cereal Killers» traemos Count Calorie, de 9 pulgadas.
-  dimoo space travel
    • Dimoo Space Travel
-  forlorn unicorn blue x ron english
    • Forlone x Ron english de 15 cm
-  frostbite superjanky by julie west
    • «Frostbite Fauna» SuperJanky mide 8 pulgadas de altura aproximadamente
-  futurama universe x
    • Colecciónalos todos:Son 16!Fry Futurama Figure – 2/24Fry (Universe B) Futurama Figure – 1/48(Extremely Rare Chase)Leela Futurama Figure – 3/48Leela (Universe B) Futurama Figure – 1/48(Extremely Rare Chase)Zoidberg (Universe B) Futurama Figure – 1/96(Super Extremely Rare Super Chase)Hedonism Bot Futurama Figure – 2/24Glow-in-the-dark Slurms Futurama Figure – 2/24Bender Futurama Figure – 2/24?? – 1/96(Mystery Super Extremely Rare Super Chase)Super King Bender Futurama Figure – 2/24Robot Devil Futurama Figure – 1/24(Rare)Roberto Futurama Figure – 2/24Calculon Futurama Figure – 2/24Glow-in-the-dark Morbo Futurama Figure – 2/24Zapp Futurama Figure with fuzzy flocked leisure suit – 2/24Glow-in-the-dark Kiff Futurama Figure – 2/24
-  garth real tree superjanky x ricardo cavolo"""}
DOCUMENT2 = {
    "title": "Art Toys",
     "content":"""    • El estilo de arte popular moderno característico del legendario artista Ricardo Cavolo se tradujo en 8 pulgadas de vinilo perfecto. «Garth Real Tree» SuperJanky es brillantemente audaz y extremadamente rudo; con ojos cuádruples, llamas ardientes, zapatillas fangosas y más tatuajes faciales que Post Malone.
- good 4 nothing
    • Limitado a 800 piezas en todo el mundo, Kidrobot nos trae esta hermosa pieza a cargo de 64 Colors.
-  gorillaz miniserie
    • GORILLAZ are BACK with their first ever BLIND BOX COLLECTION. 2D, Murdoc, Russel, Noodle and their extended family from Gorillaz world including Bonesy, Pazuzu and more are here.  Secure all 18 PLUS, alternate dimensions deserve a few alternate colorways, wouldn’t you agreeEnvíos a partir de febrero
-  gorillaz space set
    • Gorillaz are black!2D, Murdoc, Noodle & Russel are prepared for ‘Strange Timez’ in spacesuits and jetpacks 🚀  with LIGHT UP EYES and special gravity defying tools for Murdoc and Noodle.medida 12 pulgadas
- homero buddha enamel pin
    • Kidrobot presenta este pin coleccionable de los Simpsons , de 1.5 pulgadas.
- homero buddha plush 10 pulgadas
    • En su eterna búsqueda de la iluminación, Homer Buddha ha ascendido al siguiente nivel. 
Kidrobot amplía su colección de Simpsons, que siempre está creciendo y que todo lo sabe, 
con un pretzel en una mano y cuentas de oración en la otra. Desde lo alto de su rosquilla
rociada, Homer no solo está listo para otra cerveza Duff, sino también para una ubicación
destacada dentro del altar de su colección Kidrobot x The Simpsons.
- homero grin x ron english"""}
DOCUMENT3 = {
    "title": "Art Toys",
     "content": """ • Ron English ha creado esta figura de 3 pulgadas Homer Grin, como parte de la adquisición de la licencia de Los Simpsons x Kidrobot.
-  honey butt the obese bee x ron english
    • De la serie de Ron English «Cereal Killers» traemos Honey Butt the Obese Bee, de 9 pulgadas.
-  kranky serie 1
    • Nuestros 14 artistas favoritos colaboraron con Janky & Guggimon para empacar sus hermosas mentes en nuestros juguetes más pequeños3.5″ de vinilo suaveEdición limitada y nunca sabes lo que vas a conseguirArtistas: 123Klan, ADD FUEL, CHEO, Clogtwo, CRAOLA, Dalek, DELVS, Flying Fortress, MIST, REDS, Scribe, Sket One, TooFly… and ???
-  lucy curious dark harbour by brandt peters
    • Kidrobot junto Kathie Olivas y Brandt Peters nos traen Lucy Curious Dark Harbor Art Figure. Mide 
8 pulgadas de altura.
-  mealtime sketracha dunny by sket one
    • Kidrobot presenta este dunny diseñado por Sket One con 20 pulgadas de alto.
-  medium warhol dunny x andy warhol
    • Colecciónalos todos!Andy Warhol «The Mark of the Beast» Dunny – 1/48 (rare)Andy Warhol «Mao» Dunny – 1/24Andy Warhol «Camouflage» Dunny – 1/24Andy Warhol «Campbell’s Soup Can» Dunny – 3/24Andy Warhol «Cow» Dunny – 1/24Andy Warhol «Dollar Sign» Dunny – 3/24Andy Warhol «$1.57 Giant Size» Dunny – 2/24Andy Warhol «Flowers» Dunny – 2/24Andy Warhol «Elvis» Dunny – 1/48 (rare)Andy Warhol «Banana» Dunny – 3/24Andy Warhol «Fragile Handle With Care» Dunny – 1/24Andy Warhol «Skulls» Dunny – 2/24Andy Warhol «Campbell’s Tomato Juice Box» Dunny – 3/24Andy Warhol «Gun» Dunny – 1/24
-  mini moon dreamer guggimon
    • From his little corner of Hell (with some beachfront property), Mini Moon Dreamer is here by Guggimon. 
-  mister self indulgence pink edition by nouar
    • De la serie de Ron English «Cereal Killers» traemos Mister Self Indulgence Pink, de 9 pulgadas.
-  mr watt 5 by abell octovan
"""}
DOCUMENT4 = {
    "title": "Art Toys",
     "content":"""     • La serie Mr. Watt continúa con la figura 5, mide 5 pulgadas de altura.
-  pharaoh hound mini figure
    • Presentamos a los primeros amigos caninos de Crypto! Colecciónalos todos y guarda tu cripto a salvo en la Blockchain Dimension. Edición limitada de 999 piezas, Tamaño: 3 pulgadas
-  rabbiteen
    • Disponible a partir de enero medida 8 pulgadas
-  rexegon black edition 5
    • Kidrobot presenta esta figura de 5 pulgadas diseñada por James Groman.
-  scarecrow
    • Kidrobot junto a Kathie Olivas y Brandt Peters nos traen esta figura de 8 pulgadas.
- slushii superjanky by pete fowler
    • Kidrobot presenta este janky diseñado por Pete Fowler con 8 pulgadas de altura.
-  stingy jack by brandt peters
    • Kidrobot junto a Kathie Olivas y Brandt Peters nos traen esta figura de 8 pulgadas.
-  the mandalorian artfx+ kotobukiya
    • Cazador de recompensas y héroe!Kotobukiya presenta a The Mandalorian en su nueva figura de la línea ARTFX+ en una escala de 1/10. 
-  the stormtrooper artfx+ kotobukiya
    • ¡Llegan más figuras ARTFX+ de Star Wars! Kotobukiya presenta al Imperial Stormtrooper en su nueva figura de la línea ARTFX+ en una escala de 1/10.
-  transformers optimus prime x quiccs
    • Colecciónalos todos!Optimus Prime 8″ Figure: 1/24Hot Rodimus 3″ Figure: 2/24Megatron 7″ Figure: 1/24(Extremely Rare Super Chase)Ultra Magnus 8″ Figure: 2/24Arcee 3″ Figure: 2/24Soundwave 7″ Figure: 1/24(Extremely Rare Super Chase)Starscream 7″ Figure: 1/24(Extremely Rare Super Chase)Grimlock 7″ Figure: 1/24(Extremely Rare Super Chase)Bumblebee 3″ Figure: 2/24Ravage 2.5″ Figure: 2/24Spike Witwicky 3″ Figure: 2/24Skywarp 7″ Figure: 1/24(Extremely Rare Super Chase)Shockwave 7″ Figure: 1/24(Extremely Rare Super Chase)Transformers Megatron (6.5 inch) Quiccs Vinyl Figure – 2/24
-  wonder woman x tara mcpherson
    • Limitado a 300 piezasMedida: 11 pulgadas de altura """}
DOCUMENT5 = {
    "title": "Cocina1",
    "content": """- Presenta tus canapés de una forma perfecta con estos platos Bite Sized. Hechos en cerámica. Contiene 4 platos.

- Esta tabla de quesos de madera de acacia está grabada artísticamente y tiene la forma de una fina botella de champán e incluye un cuchillo de queso de acero inoxidable dorado.

- Están hechos con silicona suave para la diversión, acero inoxidable para la función y una interfaz de usuario avanzada para que puedas tener el futuro en tus propias manos.

- Fred nos trae esta copa medidora como un homenaje a Julia Child y su frase característica: "Yo disfruto mucho cocinar con vino, tanto que a veces le pongo un poco a la comida".

- No dejes que esas molestas tapas de botella te molesten. Con este destapador, deshazte de esas tapas. La aleación de zinc de alta resistencia chapada en oro es tan duradera como deslumbrante.

- Disfruta de este vaso tequilero en forma de calavera para tu bebida favorita.

- ¿Tomas tu café por la mañana ligero o dulce, o negro y fuerte? De cualquier manera, obtendrás un puñado de tu bebida favorita con esta nueva taza. La elegante cerámica de al fuego de marfil contrasta con el mango metálico de acero.

- Medida 4 oz. Tiene espacio para escribir. Medidas: 10.5 cm (ancho) x 14.5 cm (altura) x 2.2 cm (profundidad).

- Disfruta tu bebida con estos shots en forma de gema de doble pared. Capacidad: 2.5 oz. Vidrio.

- Tómate un descanso de tu caos diario, y deja que este elefante te vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

- Tómate un descanso de tu caos diario, y deja que este conejo vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

- Tómate un descanso de tu caos diario, y deja que este llama vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

- Aviéntate a tener una mini pool party en tu taza con estos infusores Float Tea. Solamente rellena con tu té favorito la cápsula de acero inoxidable, ponlo en tu taza, relájate y disfruta.

- Aviéntate a tener una mini pool party en tu taza con estos infusores Float Tea. Solamente rellena con tu té favorito la cápsula de acero inoxidable, ponlo en tu taza, relájate y disfruta.

- Aviéntate a tener una mini pool party en tu taza con estos infusores Float Tea. Solamente rellena con tu té favorito la cápsula de acero inoxidable, ponlo en tu taza, relájate y disfruta.

- Llénalo con tu ramo de hierbas favorito y colócalo en tu caldo preferido. Está hecho de silicona segura para alimentos y aguanta temperaturas de hasta 232 grados.

- Tómate un descanso de tu caos diario, y deja que este manatí vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

- Tómate un descanso de tu caos diario, y deja que este pulpo te vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

- Tómate un descanso de tu caos diario, y deja que este perezoso vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

"""}
DOCUMENT6 = {
    "title": "Cocina2",
    "content": """- Sirve tus bocadillos con estos increíbles bigotes. Contiene 18 piezas.

- Deja que esta ballena nade alrededor de tu taza, introduce las hierbas aromáticas dentro de ella y sumérgela en tu taza con agua caliente para que disfrutes de un rico té.

- Infusor de té con forma de tiburón, hecho de silicona, en color gris. Sus mandíbulas se sujetan de tu taza mientras su cola sostiene las hojas de tu bebida favorita.

- Infusor de té con forma de tiburón, hecho de silicona, en color gris. Sus mandíbulas se sujetan de tu taza mientras su cola sostiene las hojas de tu bebida favorita. Dimensiones: 8 cm. x 9 cm. x 5 cm. aproximadamente.

- Medidas: 7 x 2.5 x 13,2 cm. Compatible para grabado. Acero inoxidable.

- Estos camaleones pegajosos te harán más fácil encontrar tu bebida. Hechos de silicona.

- Disfruta tus cereales o comida favorita en este divertido plato que da la sensación de estar derramándose.

- Cuchara/Tenedor con forma de Terodáctilo. Hecho de acero inoxidable, libre de BPA.

- Si eres de esas personas para las que las cosas deben ser perfectas, entonces esta tabla de cortar es para ti. El Obsessive Chef es una tabla de cortar de 9 x 12 pulgadas hecha de bambú fuerte y sostenible, y claramente indica las medidas más precisas en detalle exacto. Así que no te preocupes... está bien exagerar un poco. Y si quieres lavarla veinte veces después de usarla, no diremos nada.

- Tómate un descanso de tu caos diario, y deja que este caballito de mar vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

- Table Saw es el accesorio perfecto para tu cocina, se puede utilizar para cortar lechuga hasta pasteles.

- Con esta taza podrás empezar un gran día, ya que adornará tus manos cada vez que tomes de ella.

- Tazas Pantone, hechas de porcelana china, disponibles en diferentes colores.

- Los Tiny Prancers son llamas pequeñas pero poderosas que cuelgan del borde de tu bebida. Cada una tiene una combinación única para que no las confundas.

- Este barquito se puede convertir en un juguete para antes de la cena. Incluye un plato, un plato hondo y un vaso.

- Tómate un descanso de tu caos di

ario, y deja que este caballito de mar vuelva ideal tu taza de té. Está hecho con silicona resistente al calor.

- Tómate un descanso de tu caos diario, y deja que esta tortuga te vuelva ideal tu taza de té. Está hecho con silicona resistente al calor."""}
DOCUMENT7 = {
    "title": "Accesorios",
    "content": """
    - Kidrobot y The Andy Warhol Foundation vuelven a juntarse para crear esta hermosa decoración POP para tu casa. The Andy Warhol x Kidrobot Resin Banana Bookends elevarán a otro nivel tus repisas. Miden 9.5” de altura. Están hechas de resina y se le puede quitar la piel al plátano para darle otro toque. ©/®/™ The Andy Warhol Foundation for the Visual Arts, Inc.

- Babero con diseño de camisa de leñador. Incluye una mordedera en forma de serrucho. Libre de BPA.

- Babero con diseño de camisa de marinero. Incluye una mordedera en forma de ancla. Libre de BPA.

- Mientras buscas un regalo de bebé único para los nuevos padres, te topas con esta adorable alcancía con forma de globo y, de repente, vuelves a la habitación de tu infancia.

Es el día en que su abuela le regaló su primera alcancía como regalo navideño, diciendo esas sabias palabras: "Un centavo ahorrado es un centavo ganado".

Solo una conejita de cerámica ordinaria con orejas rosadas y una linda sonrisa, tu alcancía tenía el misterioso poder de hacer que cualquier día parezca mejor. Pero el orgullo que sentirías cada vez que arrojaste otro centavo o níquel en tu banco de monedas marcando la diferencia en ese entonces.

Entonces, aunque este conejito de globo brillante es una alcancía más elegante y moderna, sabes que tiene el mismo objetivo simple: convertirse en un recuerdo feliz de la infancia para otra persona y una decoración que querrían tener cuando eran niños, adolescentes, e incluso como adulto.

- Encuentra ahora estos diseños exclusivos de Rick & Morty en Rojo Bermelo. Unitalla. 5 Modelos diferentes.

- 4 oz. Licorera dentro de un libro. Tiene espacio para escribir en el libro. Medidas: 10.5 cm (ancho) x 14.5 cm (alto) x 2.2 cm (profundidad).
  
    """}

DOCUMENT8 = {
    "title": "Accesorios",
    "content": """ 
    - Kidrobot presenta este pin coleccionable de los Simpsons, de 1.5 pulgadas.

- Pon tus llaves en orden con este llavero en forma de diamante.

- Lonchera con forma de Robot metálico.

- Lonchera con forma de case de guitarra.

- Babero con diseño de camisa de marinero. Incluye una mordedera en forma de ancla. Libre de BPA.

- ¡Lleva un compañero de cupcakes a donde quiera que vayas! Llavero de esmalte de doble cara.

- ¡Adjunta un unicornio a tus llaves y agrega instantáneamente un poco de magia a tus aventuras! Llavero de esmalte de doble cara.

- ¡Adjunta galaxias de bondad a tus llaves! Siente tu polvo de estrellas brillar. Llavero de esmalte de doble cara.

- Adecuado para bebés de 0 meses a 6 meses. El chupón está hecho de un material suave y flexible. Libre de BPA.

- Cambia de color cuando se moja. El patrón blanco se cambia a diferentes colores cuando se moja. Mecanismo de alta calidad para abrir y sostener el paraguas. 100 cm de diámetro cuando está abierto. Se entrega a partir del 23 de septiembre.

- Nuevos pines de Momiji disponibles.

- Nuevos pines de Momiji disponibles.

- Colecciónalos todos! Homer Simpson Enamel Pin – 2/20, Marge Simpson Enamel Pin – 2/20, Bart Simpson Enamel Pin – 3/20, Lisa Simpson Enamel Pin – 1/20, Maggie Simpson Enamel Pin – 1/20, Scratchy Enamel Pin – 1/20, Itchy Enamel Pin – 1/20, Duffman Enamel Pin – 2/20, Barney Enamel Pin – 1/20, Funzo Enamel Pin – 1/40, Snake Enamel Pin – 3/40, Apu Enamel Pin – 2/20, Devil Flanders Enamel Pin – 1/40.

- Stashed son perfectos para guardar tus cosas de una manera genial. Contiene 3 piezas. 
    """}

DOCUMENT9 = {
    "title": "Saludo",
    "content": """
    Biemvenido a Rojo Bermelo. Soy un asistente virtual y estoy encantando de poder ayudarte, tenemos una gran varidad de Art Toys, Artículos para Cocina y Accesorios
    """}

DOCUMENT10 = {
    "title": "Información de Contacto",
    "content": """
    No contamos con tienda física. Si tienes dudas, no dudes en contactarnos al Teléfono: 55 6586 6307. Correo electrónico: hola@rojobermelo.mx
    """}

DOCUMENT10 = {
    "title": "Información de la tienda Rojo Bermelo",
    "content": """
    No contamos con tienda física. Si tienes dudas, no dudes en contactarnos al Teléfono: 55 6586 6307. Correo electrónico: hola@rojobermelo.mx
    """}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3,DOCUMENT4,DOCUMENT5,DOCUMENT6,DOCUMENT7,DOCUMENT8,DOCUMENT9,DOCUMENT10]