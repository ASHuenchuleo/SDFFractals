import { Component, AfterViewInit } from '@angular/core';
import * as THREE from 'three';
import Stats from 'stats.js';
import {GUI} from 'dat.gui';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
const vertexShader = require('raw-loader!glslify-loader!./vertex.glsl');
const fragmentShader = require('raw-loader!glslify-loader!./fragment.glsl');



const scenes = {
  SIMPLE: 1,
  INFINITE: 2,
  INTERSECTION: 3,
  LIGHTBULB: 4,
  PYRAMID: 5,
  SPONGE: 6,
  MANDELBULB: 7
};

@Component({
  selector: 'app-marching-view',
  templateUrl: './marching-view.component.html',
  styleUrls: ['./marching-view.component.css']
})
export class MarchingViewComponent implements AfterViewInit{
  cube: THREE.Mesh;
  material: THREE.ShaderMaterial;

  /** width of the canvas */
  protected width = window.innerWidth - 50;
  /** height of the canvas */
  protected height = window.innerHeight - 50;
  /** configs */
  protected maxIters = 32;
  protected worldScale = 1.;
  protected tolerance = 1e-7;
  protected stepMultiplier = 1.0;
  protected tiled = false;
  protected displace = false;
  protected drawDistance = 500.;
  protected isAnimating = true;
  protected shadows = true;
  /** Scene for the view */
  protected scene;
  /** Camera of the view */
  protected camera;
  protected viewCamera;
  /** Renderer of the scene */
  protected renderer;
  /** Camera control */
  protected controls;

  /** Name of the div where it should be drawn */
  protected divName = 'view-div';

  protected aspect;

  protected sceneType;

  stats: Stats = new Stats();
  clock: THREE.Clock = new THREE.Clock();

  constructor() {
  }

  initScene(): void
  {
    const elem = document.getElementById(this.divName);

    /* Set up renderer */
    this.renderer = new THREE.WebGLRenderer({alpha : true});
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio( window.devicePixelRatio );
    elem.appendChild(this.renderer.domElement);


    /* Set up camera */
    this.aspect = this.width / this.height;
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1 / Math.pow( 2, 53 ), 100);
    this.camera.position.z = 1;
    this.viewCamera = new THREE.PerspectiveCamera( 60, 1, 0.1, 1 );
    this.viewCamera.position.x = 100;
    this.controls = new OrbitControls(this.viewCamera, this.renderer.domElement);

    /* Set up scene */
    this.scene = new THREE.Scene();

    // Lights
    const light1 = new THREE.PointLight(0xff0000);
    light1.position.set( -5, 0, 5 );
    this.scene.add( light1 );

    this.stats.domElement.style.position = 'absolute';
    this.stats.domElement.style.bottom = '0px';
    document.body.appendChild( this.stats.domElement );

    // Windows resize
    const onWindowResize = () => {


      this.aspect = this.width / this.height;

      this.camera.updateProjectionMatrix();

      this.renderer.setSize(this.width, this.height);

    };
    window.addEventListener( 'resize', onWindowResize, false );
  }

  initInteractive(): void {
    const gui = new GUI();
    const sceneFolder = gui.addFolder('Scene');
    const geoController = sceneFolder.add({Geometry: 'simple'}, 'Geometry',
      [ 'union', 'intersection', 'infinite', 'lightbulb', 'pyramid', 'sponge', 'mandelbulb' ] );
    geoController.onChange((type) => {
      switch (type){
        case 'union':
          this.sceneType = scenes.SIMPLE;
          break;
        case 'infinite':
          this.sceneType = scenes.INFINITE;
          break;
        case 'intersection':
          this.sceneType = scenes.INTERSECTION;
          break;
        case 'lightbulb' :
          this.sceneType = scenes.LIGHTBULB;
          break;
        case 'pyramid' :
          this.sceneType = scenes.PYRAMID;
          break;
        case 'sponge' :
          this.sceneType = scenes.SPONGE;
          break;
        case 'mandelbulb' :
          this.sceneType = scenes.MANDELBULB;
          break;
        default:
          break;
      }
    });

    const itersController = sceneFolder.add(this, 'maxIters', 2, 256);
    const toleranceController = sceneFolder.add(this, 'tolerance');
    const drawDistanceController = sceneFolder.add(this, 'drawDistance', 1., 2000.);
    const stepController = sceneFolder.add(this, 'stepMultiplier', 0.1, 1.);
    const worldScaleController = sceneFolder.add(this, 'worldScale', .01, 5.);

    const tiledController = sceneFolder.add(this, 'tiled');
    const animatedController = sceneFolder.add(this, 'isAnimating');
    const displaceController = sceneFolder.add(this, 'displace');
    const shadowsController = sceneFolder.add(this, 'shadows');

  }


  ngAfterViewInit(): void {
    this.initScene();
    this.initInteractive();
    let geom;
    let material;
    geom = new THREE.BufferGeometry();
    geom.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array([
                                                                  -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0
                                                                  ]), 3 ) );

    this.sceneType = scenes.SIMPLE;
    let uniforms = {
      colorB: {type: 'vec3', value: new THREE.Color(0.5 * 0.41, 0.5 * 0.05, 0.5 * 0.675)},
      colorA: {type: 'vec3', value: new THREE.Color(0, 0, 0)},
      resolution: {type: 'vec2', value: new THREE.Vector2(this.width, this.height)},
      camera: {type: 'v3', value: this.viewCamera.position},
      target: {type: 'v3', value: new THREE.Vector3(0., 0., 0.)},
      fov: {type: 'f', value: 60},
      frame: {type: 'f', value: 0},
      scene: {type: 'i', value: this.sceneType},
      maxIters: {type: 'i', value: this.maxIters},
      stepMultiplier: {type: 'f', value: this.stepMultiplier},
      tiled: {type: 'f', value: this.tiled},
      drawDistance: {type: 'f', value: this.drawDistance},
      tolerance: {type: 'f', value: this.tolerance},
      worldScale: {type: 'f', value: this.worldScale},
      displace: {type: 'b', value: this.displace},
      shadows: {type: 'b', value: this.shadows}
    };
    uniforms = THREE.UniformsUtils.merge(
      [THREE.UniformsLib.lights,
        uniforms
      ]
    );
    material = this.material = new THREE.ShaderMaterial( {
      uniforms: uniforms,
      vertexShader: vertexShader.default,
      fragmentShader: fragmentShader.default,
      lights: true

    } );
    this.cube = new THREE.Mesh(geom, material);

    this.scene.add(this.cube);

    this.animate();
  }

  animate = () => {
    this.stats.begin();
    requestAnimationFrame( this.animate );
    const delta = this.clock.getDelta();

    this.renderer.render( this.scene, this.camera );
    this.material.uniforms.camera.value = this.viewCamera.position;
    if(this.isAnimating) {
      this.material.uniforms.frame.value += 1.;
    }
    this.material.uniforms.maxIters.value = this.maxIters;
    this.material.uniforms.tolerance.value = this.tolerance;
    this.material.uniforms.worldScale.value = this.worldScale;
    this.material.uniforms.displace.value = this.displace;
    this.material.uniforms.stepMultiplier.value = this.stepMultiplier;
    this.material.uniforms.tiled.value = this.tiled;
    this.material.uniforms.scene.value = this.sceneType;
    this.material.uniforms.drawDistance.value = this.drawDistance;
    this.material.uniforms.shadows.value = this.shadows;

    // this.cube.rotation.x += 0.01;
    // this.cube.rotation.y += 0.01;

    this.controls.update(delta);
    // this.renderer.update(delta);
    this.stats.end();

  }
}
